import torch
from input_pipeline.wenet_datapipes.datapipes import TextLineDataPipe
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter.sharding import (
    SHARDING_PRIORITIES, ShardingFilterIterDataPipe)


@functional_datapipe('greedy_pack')
class MaxTextPackDataPipe(IterDataPipe):
    """
    Perform greedy sequence packing and batching
    Example:
         Two input examples get combined to form an output example.
         The input examples are:
         {"input": [8, 7, 1, 0], "target":[4, 1, 0]}
         {"input": [2, 3, 4, 1], "target":[5, 6, 1]}
         The output example is:
         {
                           "input": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
              "input_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
                  "input_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                          "target": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
             "target_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
                 "target_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
         }


    """

    def __init__(self,
                 dataset: IterDataPipe,
                 max_length: int,
                 pad_value=0) -> None:
        super().__init__()

        self.dp = dataset
        self.keys = {"input", "target"}
        self.max_length = max_length
        self.pad_value = pad_value

    def __iter__(self):
        buffer = {key: [] for key in self.keys}  # 初始化缓冲区
        buffer_length = 0

        for elem in self.dp:
            assert self.keys.issubset(
                elem), f"Dataset must contain {self.keys}"

            inputs = self._rstrip(elem["input"])
            targets = self._rstrip(elem["target"])

            if buffer_length + len(
                    inputs) > self.max_length or buffer_length + len(
                        targets) > self.max_length:
                yield self._pack_sequences(buffer)
                buffer = {key: [] for key in self.keys}  # 重置缓冲区
                buffer_length = 0

            buffer["input"].append(inputs)
            buffer["target"].append(targets)
            buffer_length += max(len(inputs), len(targets))

        if buffer_length > 0:
            yield self._pack_sequences(buffer)

    def _rstrip(self, sequence):
        if not sequence:
            return sequence
        index = len(sequence)
        while index > 0 and sequence[index - 1] == self.pad_value:
            index -= 1
        return sequence[:index]

    def _pack_sequences(self, buffer):

        packed_inputs = []
        packed_targets = []
        inputs_segmentation = []
        targets_segmentation = []
        inputs_position = []
        targets_position = []

        current_segmentation_id = 1

        for input_seq, target_seq in zip(buffer["input"], buffer["target"]):
            input_len = len(input_seq)
            target_len = len(target_seq)

            packed_inputs += input_seq
            packed_targets += target_seq

            packed_inputs_segmentation = [current_segmentation_id] * input_len
            packed_targets_segmentation = [current_segmentation_id
                                           ] * target_len

            inputs_segmentation += packed_inputs_segmentation
            targets_segmentation += packed_targets_segmentation

            packed_inputs_position = list(range(input_len))
            packed_targets_position = list(range(target_len))

            inputs_position += packed_inputs_position
            targets_position += packed_targets_position

            current_segmentation_id += 1

        inputs_padding_length = self.max_length - len(packed_inputs)
        targets_padding_length = self.max_length - len(packed_targets)

        if inputs_padding_length > 0:
            packed_inputs += [self.pad_value] * inputs_padding_length
            inputs_segmentation += [self.pad_value] * inputs_padding_length
            inputs_position += [self.pad_value] * inputs_padding_length

        if targets_padding_length > 0:
            packed_targets += [self.pad_value] * targets_padding_length
            targets_segmentation += [self.pad_value] * targets_padding_length
            targets_position += [self.pad_value] * targets_padding_length

        return {
            "input":
            torch.tensor(packed_inputs[:self.max_length]),
            "input_segmentation":
            torch.tensor(inputs_segmentation[:self.max_length]),
            "input_position":
            torch.tensor(inputs_position[:self.max_length]),
            "target":
            torch.tensor(packed_targets[:self.max_length]),
            "target_segmentation":
            torch.tensor(targets_segmentation[:self.max_length]),
            "target_position":
            torch.tensor(targets_position[:self.max_length])
        }


class MaxTextWenetRawDatasetSource(IterDataPipe):

    def __init__(self,
                 filenames: str,
                 prefetch: int = 500,
                 shuffle: bool = False,
                 shuffle_size: int = 10000,
                 cycle: int = 1) -> None:
        super().__init__()
        self.dp = TextLineDataPipe(filenames)
        if shuffle:
            self.dp = self.dp.shuffle(buffer_size=shuffle_size)
        self.dp = self.dp.repeat(cycle).prefetch(prefetch)
        self.dp = self.dp.sharding_filter()

    def __iter__(self):
        for d in self.dp:
            yield d
