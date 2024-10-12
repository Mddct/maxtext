# resize embedding and lm head for audio from maxtext's checkpoint
import argparse
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import orbax
import orbax.checkpoint as ocp
from flax.training import train_state


def get_args():
    parser = argparse.ArgumentParser(
        description='resize embedding for maxtext LLM')
    parser.add_argument('--checkpoint_path',
                        required=True,
                        type=str,
                        help='checkpoint')
    parser.add_argument('--step',
                        required=True,
                        type=int,
                        help='checkpoint step to restore')
    parser.add_argument('--new_num_tokens',
                        required=True,
                        type=int,
                        help='num of new tokens to add')
    parser.add_argument('--save_to',
                        type=str,
                        required=True,
                        help='tokenizer model name')

    args = parser.parse_args()
    return args


def main(checkpoint_path, new_tokens, save_to, step=0):

    item_names = ("items", )
    item_handlers = {
        "items": ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
    }
    checkpointer = ocp.CheckpointManager(
        checkpoint_path,
        item_names=item_names,
        item_handlers=item_handlers,
    )
    state = checkpointer.restore(step=step)
    params = state['items']['params']['params']

    initializer = nn.initializers.variance_scaling(scale=1.0,
                                                   mode="fan_in",
                                                   distribution="normal",
                                                   out_axis=0)

    emb_rng, _ = jax.random.split(jax.random.PRNGKey(2024))

    embedding = params['token_embedder']['embedding']
    print(embedding, embedding.shape)
    vocab_size, dim = embedding.shape
    new_vocab_size = new_tokens + vocab_size
    new_vocab_table = jnp.zeros((new_vocab_size, dim))
    new_vocab_table = new_vocab_table.at[:vocab_size, :].set(embedding)
    new_vocab_table = new_vocab_table.at[vocab_size:, :].set(
        initializer(emb_rng, (new_tokens, dim)))

    params['token_embedder']['embedding'] = new_vocab_table
    print(params['token_embedder']['embedding'],
          params['token_embedder']['embedding'].shape)

    state_new = train_state.TrainState(
        step=0,
        apply_fn=None,
        params={"params": params},
        tx=None,
        opt_state={}  # type: ignore
    )

    item_handlers = {
        "items": ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
    }
    save_checkpointer = ocp.CheckpointManager(
        args.save_to,
        item_names=item_names,
        item_handlers=item_handlers,
    )

    save_checkpointer.save(
        0,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(item=state_new)),
        force=True,
    )
    save_checkpointer.wait_until_finished()


if __name__ == '__main__':
    args = get_args()
    main(args.checkpoint_path, args.new_num_tokens, args.save_to, args.step)
