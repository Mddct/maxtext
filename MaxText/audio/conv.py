from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from flax import linen as nn

# Type alias for padding type
ConvPaddingType = Union[str, Tuple[Tuple[
    int, int], ...]]  # Can be "SAME", "VALID", "CAUSAL" or a custom tuple


def conv_explicit_padding(
    *,
    window: Sequence[int],
    strides: Sequence[int],
    padding: ConvPaddingType,
    dilation: Optional[Sequence[int]] = None,
) -> ConvPaddingType:
    """
    Computes explicit padding based on padding type for 2D convolutions.

    Handles three main padding types:
    - "SAME": Pads to keep the output size the same as input, if strides are 1.
    - "VALID": No padding; output size shrinks as necessary.
    - "CAUSAL": Pads only before the start of the input for causal modeling.

    Args:
        window: Size of the convolution window (kernel size).
        strides: Stride values along each dimension.
        padding: Padding type ("SAME", "VALID", "CAUSAL") or custom padding tuple.
        dilation: Dilation rate along each dimension (optional; defaults to no dilation).

    Returns:
        Tuple of padding values for each dimension.

    Raises:
        ValueError: If unsupported padding type is provided.
    """
    # If `padding` is already a custom padding tuple, return it directly
    if not isinstance(padding, str):
        return padding

    # Default to no dilation if not specified
    if dilation is None:
        dilation = (1, ) * len(window)

    # Helper function for SAME padding calculation
    def same_padding(window, dilation):
        # Compute dilated window size
        dilate_window = conv_dilate_window(window=window, dilation=dilation)
        # Total padding required to keep output size the same
        pad_total = tuple(w - 1 for w in dilate_window)
        # Split padding evenly between left and right
        pad_left = tuple(pt // 2 for pt in pad_total)
        pad_right = tuple(pt - pl for pt, pl in zip(pad_total, pad_left))
        # Return as tuple of (left, right) padding pairs for each dimension
        return tuple(zip(pad_left, pad_right))

    # Calculate padding for "SAME" mode
    if padding == "SAME":
        return same_padding(window, dilation)

    # "VALID" mode has no padding (shrinks output as necessary)
    elif padding == "VALID":
        return ((0, 0), ) * len(window)

    # Calculate padding for "CAUSAL" mode (causal padding only applies to the time dimension)
    elif padding == "CAUSAL":
        # Only applies causal padding to the first (time) dimension
        dilate_window = conv_dilate_window(window=window[:1],
                                           dilation=dilation[:1])[0]
        dilate_stride = strides[0] * dilation[0]
        # Calculate padding required before (left) and after (right)
        pad_left = dilate_window - dilate_stride
        pad_right = dilate_stride - 1
        assert pad_left + pad_right == dilate_window - 1, "Padding mismatch in CAUSAL mode"

        # Tuple containing causal padding for the time dimension
        causal_padding = ((pad_left, pad_right), )

        # Apply "SAME" padding for remaining dimensions if present
        if len(window) > 1:
            causal_padding += same_padding(window[1:], dilation[1:])

        return causal_padding

    # Raise error for unsupported padding types
    else:
        raise ValueError(f"{padding} padding is not supported.")


def conv_dilate_window(
        *,
        window: Sequence[int],
        dilation: Optional[Sequence[int]] = None) -> Tuple[int, ...]:
    """
    Calculates the effective window size after applying dilation to the convolution window.

    Args:
        window: The original convolution window size for each dimension.
        dilation: The dilation rate for each dimension, which specifies the spacing between kernel elements.

    Returns:
        A tuple containing the effective window size for each dimension after applying dilation.
    """
    # If no dilation is specified or if dilation is 1 for all dimensions, return the original window
    if dilation is None or all(d == 1 for d in dilation):
        return tuple(window)

    # Calculate the effective window size for each dimension with dilation
    # Formula: effective_size = 1 + (window_size - 1) * dilation
    return tuple(1 + d * (w - 1) for w, d in zip(window, dilation))


def compute_conv_paddings(
    in_paddings: jnp.ndarray,
    *,
    window: int,
    stride: int,
    conv_padding: ConvPaddingType,
    anchor: Optional[int] = None,
) -> jnp.ndarray:
    """
    Compute output paddings based on convolution parameters and the padding configuration.

    The output padding value is determined by the padding at the anchor point within the convolution
    window. If `anchor` is None, it defaults to the left padding position.

    Args:
        in_paddings: Tensor of shape [batch_size, seq_len], where 0 indicates valid and 1 indicates padding.
        window: Convolution window size along the time axis.
        stride: Convolution stride size along the time axis.
        conv_padding: Padding type; can be "SAME", "VALID", "CAUSAL", or a tuple specifying (left, right) padding.
        anchor: Optional integer specifying the anchor position within the window. Determines validity of each window.
                If None, defaults to left_time_padding.

    Returns:
        out_paddings: Tensor of shape [batch_size, seq_len] indicating output padding.

    Raises:
        ValueError: If `anchor` is outside the range of `[left_time_padding, window - right_time_padding)`.
    """
    chex.assert_rank(in_paddings,
                     2)  # Ensure the input paddings tensor is 2-dimensional

    # Get the explicit padding values (left and right) based on the padding type
    conv_padding = conv_explicit_padding(window=(window, ),
                                         strides=(stride, ),
                                         padding=conv_padding)
    window = conv_dilate_window(
        window=(window, ))[0]  # Adjust window if dilated
    left_pad, right_pad = conv_padding[0]  # Left and right padding
    pad_total = window - 1  # Total padding size

    # Determine the anchor point within the window for padding calculation
    if anchor is None:
        anchor = left_pad  # Default to left pad if anchor is not specified
    elif not left_pad <= anchor < window - right_pad:
        raise ValueError(
            f"anchor ({anchor}) must be in range [{left_pad}, {window - right_pad})."
        )

    # Calculate the starting index in the input sequence based on the anchor and padding
    start_index = anchor - left_pad
    valid_window = pad_total - left_pad - right_pad
    valid_window_right_pad = valid_window - start_index
    seq_len = in_paddings.shape[1]  # Length of the input sequence

    # Calculate the limit index for slicing based on valid window size and stride
    limit_index = max(seq_len - valid_window_right_pad, start_index)
    if seq_len < start_index:  # Adjust start and limit indices if sequence is too short
        start_index = 0
        limit_index = 0

    # Perform slicing along the time dimension, applying stride
    out_paddings = jax.lax.slice_in_dim(in_paddings,
                                        start_index=start_index,
                                        limit_index=limit_index,
                                        stride=stride,
                                        axis=1)

    return out_paddings


class Conv2DWith1DPadding(nn.Module):
    """2D convolution with 1D padding along the time axis, designed for audio inputs.

    This layer performs a 2D convolution but applies padding only along the time axis.
    Inputs and outputs are in NHWC format: [batch, time, frequency, channels].
    """
    output_dim: int  # Number of output channels
    window: Tuple[int, int] = (3, 3)  # Convolution kernel size (height, width)
    strides: Tuple[int, int] = (2, 2)  # Stride for convolution along each axis
    padding: Tuple[Tuple[int, int], Tuple[int, int]] = (
        (1, 1), (1, 1))  # Padding for time and frequency

    """ https://github.com/apple/axlearn/blob/main/axlearn/common/layers.py#L1186
    For examples with window=5,
        1. "SAME" padding case,
            * padding=(2,2): (0 0 0 0 0)
            * anchor index is 2: (0 0 |0| 0 0)
                        pad  |           | pad
            paddings:     0 0|0 0 0 1 1 1|1 1
                          |___0___|
                            |___0___|
                              |___0___|
                                |___1___|
                                  |___1___|
                                    |___1___|

        2. "VALID" padding case,
            * padding=(0,0): (0 0 0 0 0)
            * anchor index is 0:  (|0| 0 0 0 0)
                    pad |           | pad
            paddings:   |0 0 0 1 1 1|
                        |0_______|
                          |0_______|

        3. The legacy "VALID" padding case,
            * padding=(0,0) and anchor=4: (0 0 0 0 0)
            * anchor index is 4:  (0 0 0 0 |0|)
                    pad |           | pad
            paddings:   |0 0 0 1 1 1|
                        |________1|
                          |________1|

        4. "CAUSAL" padding case,
            * padding=(4,0): (0 0 0 0 0)
            * anchor index is 4:  (0 0 0 0 |0|)
                        pad      |           | pad
            paddings:     0 0 0 0|0 0 0 1 1 1|
                          |_______0|
                            |_______0|
                              |_______0|
                                |_______1|
                                  |_______1|
                                    |_______1|

        5. "CAUSAL" with lookahead=1,
            * padding=(3, 1): (0 0 0 0 0)
            * anchor index is 3:  (0 0 0 |0| 0)
                        pad    |           | pad
            paddings:     0 0 0|0 0 0 1 1 1|1
                          |_____0_|
                            |_____0_|
                              |_____0_|
                                |_____1_|
                                  |_____1_|
                                    |_____1_|

        6. Arbitrary padding case,
            * padding=(2,1): (0 0 0 0 0)
            * anchor index is 2:  (0 0 |0| 0 0)
                        pad  |           | pad
            paddings:     0 0|0 0 0 1 1 1|1
                          |___0___|
                            |___0___|
                              |___0___|
                                |___1___|
                                  |___1___|
        """

    """
    anchor: Optional[
        int] = None  # Anchor point for determining output validity

    @nn.compact
    def __call__(self, x: jnp.ndarray,
                 paddings: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            x: Input tensor with shape [batch, time, frequency, input_channels].
            paddings: Padding tensor with shape [batch, time], where 0 indicates valid and 1 indicates padding.

        Returns:
            output: The convolved output tensor with shape [batch, time, frequency, output_dim].
            output_paddings: Updated padding tensor with shape [batch, time].
        """
        # Define the Conv2D layer
        conv = nn.Conv(
            features=self.output_dim,
            kernel_size=self.window,
            strides=self.strides,
            padding=self.padding  # Set padding as needed
        )

        # Mask the input using the paddings to zero-out invalid time steps
        x = x * (1 - paddings[..., None, None])

        # Apply the 2D convolution to the masked input
        output = conv(x)

        # Compute output paddings based on the convolution's stride, padding, and window size
        output_paddings = compute_conv_paddings(in_paddings=paddings,
                                                window=self.window[0],
                                                stride=self.strides[0],
                                                conv_padding=self.padding,
                                                anchor=self.anchor)

        # Apply the computed paddings to the convolution output
        output = output * (1 - output_paddings[..., None, None])

        return output, output_paddings
