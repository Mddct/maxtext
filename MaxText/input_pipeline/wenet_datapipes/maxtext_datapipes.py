import torch
from input_pipeline.wenet_datapipes.datapipes import TextLineDataPipe
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter.sharding import (
    SHARDING_PRIORITIES, ShardingFilterIterDataPipe)


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
