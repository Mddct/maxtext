"""
Copyright (c) 2024 Wenet Community. (authors: Dinghao Zhou)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Optional, Tuple

import common_types
import flax.linen as nn
import jax
import jax.numpy as jnp
from common_types import BATCH
from layers.initializers import nd_dense_init
from layers.linears import DenseGeneral

Config = Any


def uniform_sqrt_init_fn(scale: float = 1.0, dtype: jnp.dtype = jnp.float_):

    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return jax.random.uniform(key,
                                  shape=shape,
                                  dtype=dtype,
                                  minval=-scale,
                                  maxval=scale)

    return init


def compute_code_histogram(onehots: jax.Array):
    """Computes histograms of the quantized codes over the codebook vocabulary.

    Args:
        onehots: Quantized onehots. Array of shape [..., num_groups, codebook_size].

    Returns:
        Histogram of the quantized codes of shape [num_groups, codebook_size].
    """
    histogram = jnp.sum(onehots, axis=tuple(range(onehots.ndim - 2)))
    return histogram


def compute_code_pplx(onehots: jax.Array, paddings: jax.Array):
    """Computes pplx and entropy of the quantized codes distribution."""
    histogram = compute_code_histogram(onehots)
    normalizer = jnp.sum(1 - paddings)
    # [num_groups, codebook_size].
    probs = histogram / jnp.maximum(normalizer, 1.0)
    log_probs = jnp.log(jnp.maximum(1.0e-30, probs))
    # [num_groups].
    sum_plogp = jnp.sum(log_probs * probs, axis=-1)
    pplx = jnp.mean(jnp.exp(-sum_plogp))
    entropy = jnp.log(pplx)
    return pplx, entropy


def compute_code_coverage(onehots: jax.Array):
    """Computes codebook coverage."""
    codebook_size = onehots.shape[-1]
    # [num_groups, codebook_size].
    histogram = compute_code_histogram(onehots)
    avg_num_covered_words = jnp.mean(
        jnp.sum((histogram > 0).astype(jnp.float32), axis=-1))
    return avg_num_covered_words / codebook_size


def quantize_vector(
        latent: jax.Array,
        codebook: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Vector quantization.
    https://github.com/wenet-e2e/wenet/blob/main/wenet/ssl/bestrq/bestrq_model.py
    Symbols in comments:
    B: batch_size.
    D: latent_dim.
    C: num_latent_classes per group
    G: num of codebook groups.

    Args:
        latent:   [B, D]
        codebook: [C, G, D // G]

    Returns:
        (quantized, codes, onehot).
        - quantized: [B, D]
        - codes:     [B, G]
        - onehot:    [B, G, C]
    """
    assert len(codebook.shape) == 3
    b, d = latent.shape
    c, g = codebook.shape[:2]
    assert d % g == 0

    latent = jnp.reshape(latent, [b, g, d // g])

    # [B, G, C]
    distance = (
        # [b, g, 1]
        jnp.sum(latent**2, -1, keepdims=True) -
        # [b, g, c]
        2 * jnp.einsum('bgd,cgd->bgc', latent, codebook) +
        # [1, g, c]
        jnp.sum(jnp.transpose(codebook, [2, 1, 0])**2, 0, keepdims=True))

    # [B, G]
    codes = jnp.argmin(distance, axis=-1)

    # [B, G, C]
    one_hot = jax.nn.one_hot(codes, c, axis=-1, dtype=jnp.float32)
    quantized = jnp.einsum('bgc,cgd->bgd', one_hot, codebook)
    quantized = jnp.reshape(quantized, [b, d])
    return quantized, codes, one_hot


def _l2_normalize(x: jax.Array,
                  axis: int,
                  epsilon: float = 1e-12) -> jax.Array:
    return x / jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=True) + epsilon)


BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
DIM = common_types.EMBED
GROUPS = 'code_groups'
CODEBOOKS = 'code_books'


class RandomVectorQuantizer(nn.Module):
    config: Any  #: Config
    input_dim: int
    num_groups: int
    num_codebooks: int
    codebook_dim: int
    normalize_inputs: bool = True
    codebook_init_std: float = 1.0
    dtype: Any = jnp.float32
    weight_dtype: Any = jnp.float32

    normalize_codebook: bool = True
    normalize_latent_vector: bool = False
    normalize_latent_per_group: bool = True

    axis_names = (BATCH, LENGTH, DIM)
    kernel_axes = (CODEBOOKS, GROUPS, DIM)

    @nn.compact
    def __call__(self, inputs, paddings):

        # Initialize the random projection layer: xavier_uniform
        random_proj_init = nd_dense_init(
            1.0,
            mode='fan_avg',
            distribution='uniform',
        )
        rand_proj = DenseGeneral(
            axis=-1,
            name='random_projection',
            features=(self.num_groups * self.codebook_dim),
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=random_proj_init,
            kernel_axes=('embed', 'mlp'),
            quant=None,
            use_bias=False,
            # matmul_precision=self.config.matmul_precision,
            matmul_precision='default',
        )
        # Initialize and freeze the codebook
        # Sect 3.1 https://arxiv.org/pdf/2202.01855.pdf.
        # Codebook uses standard Gaussian initialization.
        random_codebook_init = nn.initializers.normal(stddev=1.0,
                                                      dtype=self.weight_dtype)
        codebook = self.param(
            'codebooks',
            nn.with_logical_partitioning(random_codebook_init,
                                         self.kernel_axes),
            (self.num_codebooks, self.num_groups, self.codebook_dim))

        # Get codebook
        if self.normalize_codebook:
            codebook = _l2_normalize(codebook, axis=-1)
        # Compute random projection
        inputs = jnp.asarray(inputs, self.dtype)
        inputs = rand_proj(
            inputs)  # [batch_size, seq_len, num_codebooks * codebook_dim]
        if self.normalize_latent_vector and not self.normalize_latent_per_group:
            inputs = _l2_normalize(inputs, -1)

        inputs = nn.with_logical_constraint(inputs, self.axis_names)
        b, l, d = inputs.shape

        # Reshape for group-wise quantization
        proj_by_group = inputs.reshape(b * l, d)
        if self.normalize_latent_vector and self.normalize_latent_per_group:
            proj_by_group = _l2_normalize(proj_by_group, -1)

        # Quantize using vector quantization
        q, codes, onehots = quantize_vector(proj_by_group, codebook)
        q = q.reshape(b, l, -1)
        codes = codes.reshape(b, l, self.num_groups)
        onehots = onehots.reshape(b, l, self.num_groups, self.num_codebooks)

        q = nn.with_logical_constraint(q, (BATCH, DIM))
        codes = nn.with_logical_constraint(codes, (BATCH, GROUPS))
        onehots = nn.with_logical_constraint(onehots,
                                             (BATCH, GROUPS, CODEBOOKS))

        # Apply paddings
        q = q * (1 - paddings)[..., None]
        codes = codes * (1 - paddings)[..., None]
        onehots = onehots * (1 - paddings)[..., None, None]

        pplx, entropy = compute_code_pplx(onehots, paddings)
        self.sow('codebook', 'coverage', compute_code_coverage(onehots))
        self.sow('codebook', 'pplx', pplx)
        self.sow('codebook', 'entropy', entropy)

        return {
            "ids":
            jax.lax.stop_gradient(codes),
            "onehots":
            jax.lax.stop_gradient(onehots),
            "quantized_vectors":
            jax.lax.stop_gradient(
                q.reshape(b, l, self.num_groups, self.codebook_dim)),
        }


class SeqVectorQuantizer(nn.Module):
    """Vector quantizer using MSE loss."""
    num_codebooks: int
    codebook_dim: int
    beta: float
    num_groups: int = 1
    normalize_codebook: bool = False
    normalize_inputs: bool = False
    dtype: jnp.dtype = jnp.float32
    weight_dtype: jnp.dtype = jnp.float32

    kernel_axes = (CODEBOOKS, GROUPS, DIM)

    def loss(self, input, quantized):
        pass

    @nn.compact
    def __call__(self, inputs: jax.Array, paddings: jax.Array) -> Any:
        """Forward function for quantization and loss calculation.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, input_dim]
            paddings: 0/1 tensor of shape [batch_size, seq_len]

        Returns:
            Dictionary with quantized vectors and losses
        """
        # Define input dimensions
        input_dim = self.num_groups * self.codebook_dim
        batch_size, seq_len = inputs.shape[:2]

        # Verify input dimension compatibility
        if inputs.shape[-1] != input_dim:
            raise ValueError(
                f"Input feature dimension must match dimensions of all codebooks."
                f"{inputs.shape[-1]} != {self.num_groups} x {self.codebook_dim}."
            )

        inputs = jnp.asarray(inputs, dtype=self.dtype)
        # uniform init
        codebook_init = nd_dense_init(1.0,
                                      mode='fan_in',
                                      distribution='uniform')
        codebook = self.param(
            'codebooks',
            nn.with_logical_partitioning(codebook_init, self.kernel_axes),
            (self.num_codebooks, self.num_groups, self.codebook_dim),
            in_axis=(0, 1),
            out_axis=(2),
            dtype=self.dtype,
        )

        # Reshape inputs by grouping according to codebooks
        inputs_by_group = jnp.reshape(
            inputs, [batch_size, seq_len, self.num_groups, self.codebook_dim])

        if self.normalize_codebook:
            codebook = _l2_normalize(codebook, -1)
        if self.normalize_inputs:
            inputs_by_group = _l2_normalize(inputs_by_group, axis=-1)

        q, codes, onehots = quantize_vector(
            inputs_by_group.reshape(-1, self.codebook_dim), codebook)
        q = q.reshape(batch_size, seq_len, -1)
        codes = codes.reshape(batch_size, seq_len, self.num_groups)
        onehots = onehots.reshape(batch_size, seq_len, self.num_groups,
                                  self.num_codebooks)
        q = nn.with_logical_constraint(q, (BATCH, DIM))
        codes = nn.with_logical_constraint(codes, (BATCH, GROUPS))
        onehots = nn.with_logical_constraint(onehots,
                                             (BATCH, GROUPS, CODEBOOKS))

        # Apply paddings
        q = q * (1 - paddings)[..., None]
        codes = codes * (1 - paddings)[..., None]
        onehots = onehots * (1 - paddings)[..., None, None]
        pplx, entropy = compute_code_pplx(onehots, paddings)

        self.sow('codebook', 'coverage', compute_code_coverage(onehots))
        self.sow('codebook', 'pplx', pplx)
        self.sow('codebook', 'entropy', entropy)

        # Calculate mean squared error loss
        num_frames = jnp.sum(1 - paddings)
        denominator = jnp.maximum(num_frames * input_dim, 1e-6)

        inputs_to_loss = (jnp.reshape(inputs_by_group,
                                      [batch_size, seq_len, -1])
                          if self.normalize_inputs else inputs)

        kmeans_loss = (jnp.sum((q - jax.lax.stop_gradient(inputs_to_loss))**2 *
                               (1 - paddings)[:, :, None]) / denominator)

        commitment_loss = (jnp.sum(
            (inputs_to_loss - jax.lax.stop_gradient(q))**2 *
            (1 - paddings)[:, :, None]) / denominator)

        self.sow('loss', 'kmeans_loss', kmeans_loss)
        self.sow('loss', 'commitment_loss', commitment_loss)
        total_loss = kmeans_loss + self.beta * commitment_loss

        # Straight-through estimator for quantized vectors
        quantized_vectors = inputs + jax.lax.stop_gradient(q - inputs)
        quantized_vectors = quantized_vectors * (1 - paddings)[:, :, None]

        outputs = {
            "ids":
            codes,
            "onehots":
            onehots,
            "quantized_vectors":
            quantized_vectors.reshape(batch_size, seq_len, self.num_groups,
                                      self.codebook_dim),
            "loss":
            total_loss,
        }

        return outputs


def gumbel_scheduler(step: jax.Array,
                     max_gumbel_temperature: float = 2.0,
                     gumbel_temperature_decay: float = 0.999995,
                     min_gumbel_temperature: float = 0.1):
    gumbel_temperature = jnp.clip(
        max_gumbel_temperature * gumbel_temperature_decay**step,
        a_min=min_gumbel_temperature,
    )
    return gumbel_temperature


class GumbelSoftmaxVectorQuantizer(nn.Module):
    """Vector quantizer using the Gumbel softmax trick.
    https://arxiv.org/pdf/1611.01144.pdf
    """
    # input_dim: int
    num_codebooks: int
    codebook_dim: int
    num_groups: int
    weight_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.float32

    kernel_axes = (CODEBOOKS, GROUPS, DIM)

    @nn.compact
    def __call__(self,
                 inputs: jax.Array,
                 paddings: jax.Array,
                 training: bool,
                 gumbel_temperature: Optional[jax.Array] = None) -> Any:
        """Forward pass for quantization using Gumbel softmax trick.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, input_dim].
            paddings: 0/1 tensor of shape [batch_size, seq_len].
            is_training: Boolean indicating if the model is in training mode.

        Returns:
            Dictionary containing quantized vectors and other outputs.
        """
        # Project inputs to logits for Gumbel-Softmax quantization
        proj_init = nd_dense_init(
            1.0,
            mode='fan_avg',
            distribution='truncated_normal',
        )
        input_proj = DenseGeneral(
            axis=-1,
            name='projection',
            features=(self.num_groups * self.num_codebooks),
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=proj_init,
            kernel_axes=('embed', 'mlp'),
            quant=None,
            use_bias=False,
            # matmul_precision=self.config.matmul_precision,
            matmul_precision='default',
        )

        codebook_init = nn.initializers.uniform(scale=1.0,
                                                dtype=self.weight_dtype)
        codebook = self.param(
            'codebooks',
            nn.with_logical_partitioning(codebook_init, self.kernel_axes),
            (self.num_codebooks, self.num_groups, self.codebook_dim))

        logits = input_proj(
            inputs)  # [batch_size, seq_len, num_codebooks * num_groups]
        logits = logits.reshape(inputs.shape[0], inputs.shape[1],
                                self.num_groups, self.num_codebooks)

        if training:
            # Apply temperature scheduling for Gumbel-Softmax
            if gumbel_temperature is not None:
                assert isinstance(gumbel_temperature, jax.Array)
                tau = gumbel_temperature
            else:
                tau = 1.0
            # Add Gumbel noise for sampling in training
            gumbel_noise = jax.random.gumbel(self.make_rng("gumbel"),
                                             logits.shape)
            logits = (logits + gumbel_noise) / tau

        # Select the max index in logits as quantization ID
        ids = jnp.argmax(logits, axis=-1)  # [batch_size, seq_len, num_groups]

        if not training:
            # Direct lookup for inference
            quantized_vectors = self._lookup(ids, codebook)
            ids = ids * (1 - paddings[:, :, None])
            quantized_vectors = quantized_vectors * (1 - paddings)[:, :, None,
                                                                   None]
        else:
            # Mask padding positions
            mask = (1 - paddings)[:, :, None]
            ids = ids * mask + (-1) * (1 - mask)

            # Convert IDs to one-hot encoding for Gumbel-Softmax
            onehots = jax.nn.one_hot(
                ids, num_classes=self.num_codebooks) * mask[:, :, :, None]
            y_soft = jax.nn.softmax(logits, axis=-1) * mask[:, :, :, None]

            # Straight-through estimator: dL/dy_soft = dL/donehots
            onehots = y_soft + jax.lax.stop_gradient(onehots - y_soft)

            # Matrix multiply one-hot with codebook to get quantized vectors
            quantized_vectors = jnp.einsum(
                "...gv,vgh->...gh",
                onehots,
                codebook,
            )
            quantized_vectors = quantized_vectors * mask[:, :, :, None]

        outputs = {
            "ids": ids,
            "quantized_vectors": quantized_vectors,
        }

        if training:
            self.sow('gumbel', 'temperature', tau)
            self.sow('gumbel', 'probs', y_soft)

        return outputs

    def _lookup(self, ids: jax.Array, codebook: jax.Array) -> jax.Array:
        """Lookup function to retrieve vectors from codebook based on ids.

        Args:
            ids: Integer tensor of shape [..., num_groups] with values
                in range [0, num_codebooks).
            codebook: Tensor of shape [num_codebooks, num_groups, codebook_dim].

        Returns:
            quantized vectors.
        """
        if ids.ndim - 1 > 11:  # Ensures we are within einsum dimension limits
            raise NotImplementedError(ids.shape)

        # Create an index for the num_codebooks axis
        g_index = jnp.expand_dims(jnp.arange(ids.shape[-1]),
                                  axis=tuple(range(ids.ndim - 1)))

        # Retrieve quantized vectors by indexing into the codebook
        quantized_vectors = codebook[ids, g_index]
        return quantized_vectors
