import argparse
import logging


def get_args():
    parser = argparse.ArgumentParser(
        description='add new words for hg tokenizer')
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        help='tokenizer model name')
    parser.add_argument('--words_list',
                        default='',
                        type=str,
                        help='new words list')
    parser.add_argument('--save_to',
                        required=True,
                        type=str,
                        help='new tokenizer model')
    parser.add_argument('--add_audio_tokens',
                        action='store_true',
                        help='add audio tokens')
    parser.add_argument('--audio_min_id',
                        type=int,
                        default=0,
                        help='audio max token id')
    parser.add_argument('--audio_max_id',
                        required=True,
                        type=int,
                        default=10000,
                        help='audio max token id')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    if args.add_audio_tokens:
        AUDIO_UNIT_TOKENS = [
            "<|startofaudio|>",
            "<|endofaudio|>",
            "<|startofalign|>",
            "<|endofalign|>",
            "<|speech2text|>",
            "<|text2speech|>",
            "<|transcribe|>",
            "<|im_start|>",
            "<|im_end|>",
            *[
                f"<|audio_{i}|>"
                for i in range(args.audio_min_id, args.audio_max_id)
            ],
        ]
        tokenizer.add_tokens(AUDIO_UNIT_TOKENS)
        logging.info(f'add numbers of audio units: {len(AUDIO_UNIT_TOKENS)}')
    if args.words_list != '':
        with open(args.words_list, 'r') as f:
            words_list = []
            for line in f:
                line = line.strip('\n')
                words_list.append(line)
            tokenizer.add_tokens(words_list)
            logging.info(f'add numbers of word tokens: {len(words_list)}')

    tokenizer.save_pretrained(args.save_to)


if __name__ == '__main__':
    main()
