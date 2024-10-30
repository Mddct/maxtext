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

import flax.linen as nn
import jax
import jax.numpy as jnp


def compute_code_histogram(onehots: jax.Array, paddings: jax.Array):
    """Computes histograms of the quantized codes over the codebook vocabulary.

    Args:
        onehots: Quantized onehots. Array of shape [..., num_codebooks, codebook_size].
        paddings: paddings of the quantized codes. Array of shape [...].

    Returns:
        Histogram of the quantized codes of shape [num_codebooks, codebook_size].
    """
    onehots = onehots * (1 - paddings)[..., None, None]
    # [num_codebooks, codebook_size].
    histogram = jnp.sum(onehots, axis=tuple(range(onehots.ndim - 2)))
    return histogram


def compute_code_pplx(onehots: jax.Array, paddings: jax.Array):
    """Computes pplx and entropy of the quantized codes distribution."""
    histogram = compute_code_histogram(onehots, paddings)
    normalizer = jnp.sum(1 - paddings)
    # [num_codebooks, codebook_size].
    probs = histogram / jnp.maximum(normalizer, 1.0)
    log_probs = jnp.log(jnp.maximum(1.0e-30, probs))
    # [num_codebooks].
    sum_plogp = jnp.sum(log_probs * probs, axis=-1)
    pplx = jnp.mean(jnp.exp(-sum_plogp))
    entropy = jnp.log(pplx)
    return pplx, entropy


def compute_code_coverage(onehots: jax.Array, paddings: jax.Array):
    """Computes codebook coverage."""
    codebook_size = onehots.shape[-1]
    # [num_codebooks, codebook_size].
    histogram = compute_code_histogram(onehots, paddings)
    avg_num_covered_words = jnp.mean(
        jnp.sum((histogram > 0).astype(jnp.float32), axis=-1))
    return avg_num_covered_words / codebook_size


class RandomVectorQuantizer(nn.Module):
    input_dim: int
    num_codebooks: int
    codebook_dim: int
    normalize_codebook: bool = True
    normalize_inputs: bool = True
    codebook_init_std: float = 1.0

    def setup(self):
        # Initialize the random projection layer
        self.rand_proj = nn.Dense(features=self.num_codebooks *
                                  self.codebook_dim,
                                  use_bias=False,
                                  kernel_init=nn.initializers.xavier_uniform())
        # Initialize and freeze the codebook
        self.codebook = self.param(
            "codebook", nn.initializers.normal(stddev=self.codebook_init_std),
            (self.num_codebooks, self.codebook_dim))
        # Normalize codebook if enabled
        if self.normalize_codebook:
            self.codebook = self.codebook / (
                jnp.linalg.norm(self.codebook, axis=-1, keepdims=True) + 1e-12)

    def __call__(self, inputs, paddings):
        # Compute random projection
        inputs = self.rand_proj(
            inputs)  # [batch_size, seq_len, num_codebooks * codebook_dim]
        inputs_by_group = inputs.reshape(inputs.shape[:2] +
                                         (self.num_codebooks,
                                          self.codebook_dim))

        # Normalize inputs if enabled
        if self.normalize_inputs:
            inputs_by_group = inputs_by_group / (jnp.linalg.norm(
                inputs_by_group, axis=-1, keepdims=True) + 1e-12)

        # Choose similarity metric
        if self.normalize_codebook:
            metric = lambda x, y: jnp.einsum("...nd,cd->...nc", x, y
                                             )  # Dot product similarity
        else:
            metric = lambda x, y: -jnp.linalg.norm(x[..., None, :] - y,
                                                   axis=-1)  # L2 distance

        # Quantize by nearest neighbor
        similarities = metric(inputs_by_group, self.codebook)
        ids = jnp.argmax(similarities, axis=-1)
        onehots = jax.nn.one_hot(ids, self.codebook.shape[0])
        quantized_vectors = jnp.einsum("...nc,cd->...nd", onehots,
                                       self.codebook)

        # Apply paddings
        quantized_vectors = quantized_vectors * (1 - paddings)[..., None, None]
        onehots = onehots * (1 - paddings)[..., None]

        return {
            "ids": ids,
            "onehots": onehots,
            "quantized_vectors": quantized_vectors,
        }
