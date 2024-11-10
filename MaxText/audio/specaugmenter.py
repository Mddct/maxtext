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
""" References:
    https://arxiv.org/abs/1904.08779
    https://arxiv.org/abs/1912.05533
"""
from typing import Optional

import jax
import jax.numpy as jnp


def mask_sampler(input_lengths: jax.Array,
                 max_length: int,
                 rng: jax.random.PRNGKey,
                 max_num_masks: Optional[int] = None,
                 max_num_masks_ratio: Optional[float] = None,
                 max_mask_length: Optional[int] = None,
                 max_mask_length_ratio: Optional[float] = None) -> jax.Array:
    """Generates masks based on the provided input lengths.

    Args:
        input_lengths: An integer tensor of shape [batch_size].
        max_length: Maximum length of inputs.
        rng: Random key for generating masks.
        max_num_masks: Maximum number of masks.
        max_num_masks_ratio: Ratio for maximum number of masks.
        max_mask_length: Maximum length of each mask.
        max_mask_length_ratio: Ratio for maximum mask length.

    Returns:
        A 0/1 tensor of shape [batch_size, max_length], where 1 means masked position.
    """
    batch_size = input_lengths.shape[0]

    # Split RNG key for mask lengths and start positions
    length_key, start_key = jax.random.split(rng, num=2)

    # Define mask length limits
    max_mask_length_ratio = max_mask_length_ratio or 1.0
    max_mask_length = min(max_mask_length or max_length, max_length)
    max_num_masks = min(max_num_masks or max_length, max_length)

    # Ensure input lengths have shape [batch_size, 1]
    input_lengths = input_lengths[:, None]

    # Compute actual mask lengths and sample their values
    max_mask_length = jnp.maximum(
        jnp.minimum(max_mask_length,
                    (max_mask_length_ratio * input_lengths).astype(jnp.int32)),
        1)
    lengths = jax.random.randint(length_key,
                                 shape=(batch_size, max_num_masks),
                                 minval=0,
                                 maxval=max_mask_length + 1)

    # Sample start positions
    starts = jax.random.randint(start_key,
                                shape=(batch_size, max_num_masks),
                                minval=0,
                                maxval=(input_lengths - lengths + 1))

    # Construct mask tensor
    mask_index = jnp.arange(max_length)
    masks = (starts[..., None] <= mask_index) & (mask_index <
                                                 (starts + lengths)[..., None])

    # Apply ratio limit if set
    if max_num_masks_ratio:
        num_masks = (max_num_masks_ratio * input_lengths).astype(jnp.int32)
        masks = masks * (jnp.arange(max_num_masks) < num_masks)[..., None]

    # Aggregate across mask axis
    return jnp.max(masks, axis=1)


def spectrum_augmenter(
        inputs: jax.Array,
        paddings: jax.Array,
        rng,
        max_num_masks: Optional[int] = None,
        max_num_masks_ratio: Optional[float] = None,
        max_mask_length: Optional[int] = None,
        max_mask_length_ratio: Optional[float] = None) -> jax.Array:
    """Applies SpecAugment to inputs.

    Args:
        inputs: A tensor of shape [batch_size, num_frames, num_freq, num_channels].
        paddings: A 0/1 tensor of shape [batch_size, num_frames].
        rng: Random key for generating masks.
        max_num_masks, max_num_masks_ratio, max_mask_length, max_mask_length_ratio: Mask sampling params.

    Returns:
        A tensor of shape [batch_size, num_frames, num_freq, num_channels] with SpecAugment applied.
    """

    batch_size, num_frames, num_freq, num_channels = inputs.shape

    # Split rng for frequency and time masks
    freq_rng, time_rng = jax.random.split(rng, 2)

    # Generate frequency masks
    freq_masks = mask_sampler(input_lengths=jnp.repeat(num_freq, batch_size),
                              max_length=num_freq,
                              rng=freq_rng,
                              max_num_masks=max_num_masks,
                              max_num_masks_ratio=max_num_masks_ratio,
                              max_mask_length=max_mask_length,
                              max_mask_length_ratio=max_mask_length_ratio)

    # Generate time masks
    time_masks = mask_sampler(input_lengths=jnp.sum(1 - paddings, axis=1),
                              max_length=num_frames,
                              rng=time_rng,
                              max_num_masks=max_num_masks,
                              max_num_masks_ratio=max_num_masks_ratio,
                              max_mask_length=max_mask_length,
                              max_mask_length_ratio=max_mask_length_ratio)

    # Apply frequency and time masks
    freq_keep = 1 - freq_masks[:, None, :,
                               None]  # Shape: [batch_size, 1, num_freq, 1]
    time_keep = 1 - time_masks[:, :, None,
                               None]  # Shape: [batch_size, num_frames, 1, 1]

    # Apply masks to input
    return inputs * freq_keep * time_keep
