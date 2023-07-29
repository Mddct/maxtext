export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
RUN_NAME=${1}

STEPS=5

python3 MaxText/train.py MaxText/configs/base.yml \
    steps=${STEPS} per_device_batch_size=4 learning_rate=0.001 warmup_steps=2000 enable_profiler=true enable_checkpointing=false \
    enable_dropout=false enable_data_shuffling=false run_name=${RUN_NAME}\
    use_int8_training=true metrics_file=metrics.txt\
    remat_policy=full\
    base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue/