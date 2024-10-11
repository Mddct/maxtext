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
"""Input pipeline using wenet datapipes."""

import glob
from functools import partial
from typing import Dict, List

import jax
import ml_collections
import multihost_dataloading
import torch
from MaxText.input_pipeline._wenet_tokenizer import HuggingFaceTokenizer
from MaxText.input_pipeline.wenet_datapipes.maxtext_datapipes import \
    MaxTextWenetRawDatasetSource
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def tokenizeOp(sample, tokenizer):
    """pretrain
    """
    text = sample['text']
    tokens, ids = tokenizer.tokenize(text)
    sample['tokens'] = tokens
    sample['input'] = ids
    sample['output'] = ids
    return {
        'input': torch.tensor(ids),
        'output': torch.tensor(ids),
    }


def trim(sample, max_length):
    sample['tokens'] = sample['tokens'][:max_length]
    sample['input'] = sample['input'][:max_length]
    sample['output'] = sample['output'][:max_length]
    return sample


def shift(sample):
    input_ids = sample['input']
    output_ids = sample['output']

    sample['input'] = input_ids[:-1]
    sample['output'] = output_ids[1:]
    return sample


def get_datasets(data_file_pattern, shuffle, epoch, prefetch):
    """Load dataset from array_record files for using with grain"""
    data_files = glob.glob(data_file_pattern)
    dataset = MaxTextWenetRawDatasetSource(data_files,
                                           prefetch,
                                           shuffle,
                                           100000,
                                           cycle=epoch)
    return dataset


def padding_fn(data: List[Dict]):
    """ Padding the data into training data

        Args:
            data: List[{input_ids, output_ids}

        Returns:
            Tuple(feats, labels)
    """

    samples = data
    inputs = [sample['input_ids'] for sample in samples]
    outputs = [sample['output_ids'] for sample in samples]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    outputs = pad_sequence(outputs, batch_first=True, padding_value=0)

    batch = {
        "inputs": inputs.numpy(force=True),
        "targets": outputs.numpy(force=True),
    }
    return batch


def preprocessing_pipeline(
    global_mesh,
    dataset,
    tokenizer_path,
    global_batch_size: int,
    max_target_length: int,
    dataloading_host_index: int,
    dataloading_host_count: int,
    tokenize=True,
    add_bos=True,
    add_eos=True,
    stage2_shuffle: bool = True,
    shuffle_size: int = 1024,
    packing: bool = True,
    shift: bool = True,
    drop_remainder: bool = True,
    num_workers: int = 1,
    seed: int = 2024,
    dataloader_prefetch: int = 10,
):
    """Use grain to pre-process the dataset and return iterators"""
    assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."
    if tokenize:
        tokenizer = HuggingFaceTokenizer(model=tokenizer_path,
                                         add_bos_token=add_bos,
                                         add_eos_token=add_eos)

        dataset = dataset.map(partial(tokenizeOp, tokenizer=tokenizer))
    dataset = dataset.map(partial(trim, max_length=max_target_length + 1))
    if stage2_shuffle:
        dataset = dataset.shuffle(shuffle_size)
    if shift:
        dataset = dataset.map(shift)
    if packing:
        pass
    else:
        dataset = dataset.batch(global_batch_size // jax.process_count(),
                                drop_last=drop_remainder,
                                wrapper_class=padding_fn)

    def worker_init_fn(worker_id):
        total_wokers_in_cluster = dataloading_host_count * num_workers
        worker_id_in_cluster = worker_id + dataloading_host_index
        info = torch.utils.data.get_worker_info()
        assert info is not None
        datapipe = info.dataset
        torch.utils.data.graph_settings.apply_sharding(
            datapipe, total_wokers_in_cluster, worker_id_in_cluster)

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(dataset,
                            batch_size=None,
                            num_workers=num_workers,
                            persistent_workers=True,
                            generator=generator,
                            collate_fn=lambda batch: batch,
                            prefetch_factor=dataloader_prefetch,
                            worker_init_fn=worker_init_fn)
    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(
        dataloader, global_mesh)

    # Return multi-host jax.Array prep iterator
    return multihost_gen


def make_wenet_dataset_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
    """Load, preprocess dataset and return iterators"""
    if not hasattr(config, 'epoch'):
        epoch = 1
    else:
        epoch = config.epoch

    dataloading_host_index = process_indices.index(jax.process_index()),
    dataloading_host_count = len(process_indices),
    train_ds = get_datasets(config.wenet_train_files, True, epoch,
                            config.first_prefetch)

    train_iter = preprocessing_pipeline(
        dataset=train_ds,
        tokenizer_path=config.tokenizer_path,
        global_batch_size=config.global_batch_size_to_load,
        global_mesh=global_mesh,
        max_target_length=config.max_target_length,
        num_workers=config.wenet_worker_count,
        stage2_shuffle=config.enable_data_shuffling,
        tokenize=config.tokenize_train_data,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
    )

    if config.eval_interval > 0:
        eval_ds = get_datasets(config.wenet_train_files, False, 1,
                               config.first_prefetch)
        eval_iter = preprocessing_pipeline(
            dataset=eval_ds,
            tokenizer_path=config.tokenizer_path,
            global_batch_size=config.global_batch_size_to_load,
            global_mesh=global_mesh,
            max_target_length=config.max_target_length,
            num_workers=config.wenet_worker_count,
            stage2_shuffle=config.enable_data_shuffling,
            tokenize=config.tokenize_train_data,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
            dataloading_host_index=dataloading_host_index,
            dataloading_host_count=dataloading_host_count,
        )

    else:
        eval_iter = None
    return train_iter, eval_iter
