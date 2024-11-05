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

from typing import Any, Tuple

import common_types
import flax.linen as nn
import jax
import jax.numpy as jnp
from layers.attentions import NdInitializer
from layers.gpt3 import DenseGeneral
from layers.initializers import nd_dense_init

Config = common_types.DType


def compute_code_histogram(onehots: jax.Array):
    """Computes histograms of the quantized codes over the codebook vocabulary.

    Args:
        onehots: Quantized onehots. Array of shape [..., num_groups, codebook_size].

    Returns:
        Histogram of the quantized codes of shape [num_groups, codebook_size].
    """
    # [num_codebooks, codebook_size].
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


class RandomVectorQuantizer(nn.Module):
    config: Config
    input_dim: int
    num_groups: int
    num_codebooks: int
    codebook_dim: int
    normalize_inputs: bool = True
    codebook_init_std: float = 1.0
    kernel_axes: Tuple[str, ...] = ()
    dtype: Any = jnp.float32
    weight_dtype: Any = jnp.float32

    normalize_codebook: bool = False
    normalize_latent_vector: bool = False
    normalize_latent_per_group: bool = True

    # xavier init
    random_proj_init: NdInitializer = nd_dense_init(
        1.0, mode='fan_avg', distribution='truncated_normal')

    # noram init
    random_codebook_init: NdInitializer = nn.initializers.normal(stddev=1.0)

    def setup(self):
        # Initialize the random projection layer
        self.rand_proj = DenseGeneral(
            axis=-1,
            name='random_projection',
            features=(self.input_dim, self.num_groups * self.codebook_dim),
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.random_proj_init,
            kernel_axes=('embed', 'mlp'),
            quant=None,
            use_bias=False,
            matmul_precision=self.config.matmul_precision,
        )
        # Initialize and freeze the codebook
        kernel_axes = ('codebooks', 'code_gorups', 'codebook_dim')
        self.codebook = self.param(
            'codebooks',
            nn.with_logical_partitioning(self.random_codebook_init,
                                         kernel_axes),
            (self.num_codebooks, self.num_groups, self.codebook_dim),
            self.weight_dtype,
        )

    def __call__(self, inputs, paddings):

        # Get codebook
        codebook = self.codebook
        if self.normalize_codebook:
            codebook = _l2_normalize(self.codebook, axis=-1)
        # Compute random projection
        inputs = jnp.asarray(inputs, self.dtype)
        inputs = self.rand_proj(
            inputs)  # [batch_size, seq_len, num_codebooks * codebook_dim]

        # TODO: logical constraint

        if self.normalize_latent_vector and not self.normalize_latent_per_group:
            inputs = _l2_normalize(inputs, -1)

        # Reshape for group-wise quantization
        b, l, d = inputs.shape
        proj_by_group = inputs.reshape(b * l * d)
        if self.normalize_latent_vector and self.normalize_latent_per_group:
            proj_by_group = _l2_normalize(proj_by_group, -1)

        # Quantize using vector quantization
        q, codes, onehot = quantize_vector(proj_by_group, codebook)
        # TODO: logical constraint

        q = q.reshape(b, l, d)
        codes = codes.reshape(b, l, self.num_groups)
        onehot = onehot.reshape(b, l, self.num_groups, self.num_codebooks)

        # Apply paddings
        q = q * (1 - paddings)[..., None]
        codes = codes * (1 - paddings)[..., None]
        onehots = onehot * (1 - paddings)[..., None, None]

        pplx, entropy = compute_code_pplx(onehot, paddings)
        self.sow('codebook', 'coverage', compute_code_coverage(onehot))
        self.sow('codebook', 'pplx', pplx)
        self.sow('codebook', 'entropy', entropy)

        return {
            "ids": jax.lax.stop_gradient(codes),
            "onehots": jax.lax.stop_gradient(onehots),
            "quantized_vectors": jax.lax.stop_gradient(q),
        }
