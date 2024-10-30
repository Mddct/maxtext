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
