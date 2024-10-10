import torch
from input_pipeline.wenet_datapipes.datapipes import TextLineDataPipe
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter.sharding import (
    SHARDING_PRIORITIES, ShardingFilterIterDataPipe)


@functional_datapipe("maxtext_shard")
class ShardDataPipe(ShardingFilterIterDataPipe):

    def __init__(self, dataset: IterDataPipe, num_of_instances_in_cluster: int,
                 instance_id: int):
        super().__init__(dataset, None)
        self.dp = dataset
        self.num_of_instances = num_of_instances_in_cluster
        self.instance_id = instance_id

    def apply_sharding(self,
                       num_of_instances,
                       instance_id,
                       sharding_group=SHARDING_PRIORITIES.DEFAULT):
        if instance_id >= num_of_instances:
            raise ValueError(
                f"instance_id({instance_id}) should be smaller than num_of_instances({num_of_instances})"
            )
        if sharding_group == SHARDING_PRIORITIES.DEFAULT:
            if len(self.groups
                   ) and SHARDING_PRIORITIES.DEFAULT not in self.groups:
                raise Exception(
                    'ShardingFilter cannot mix DEFAULT and non DEFAULT groups')
        else:
            if SHARDING_PRIORITIES.DEFAULT in self.groups:
                raise Exception(
                    'ShardingFilter cannot mix DEFAULT and non DEFAULT groups')
        if self.num_of_instances is not None:
            num_of_instances = self.num_of_instances
        if self.instance_id is not None:
            instance_id = self.instance_id
        self.groups[sharding_group] = (num_of_instances, instance_id)
        self._update_num_of_instances()


class MaxTextWenetRawDatasetSource(IterDataPipe):

    def __init__(self,
                 filenames: str,
                 prefetch: int = 500,
                 shuffle: bool = False,
                 shuffle_size: int = 10000,
                 num_of_instances_in_cluster=None,
                 instance_id=None,
                 cycle: int = 1) -> None:
        super().__init__()
        self.dp = TextLineDataPipe(filenames)
        if shuffle:
            self.dp = self.dp.shuffle(buffer_size=shuffle_size)
        self.dp = self.dp.repeat(cycle).prefetch(prefetch)
        self.dp = self.dp.maxtext_shard(
            num_of_instances_in_cluster=num_of_instances_in_cluster,
            instance_id=instance_id)

    def __iter__(self):
        for d in self.dp:
            yield d
