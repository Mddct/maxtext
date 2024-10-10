from abc import ABC, abstractmethod, abstractproperty
from os import PathLike
from typing import Dict, List, Tuple, Union

T = Union[str, bytes]
Type = T


class BaseTokenizer(ABC):

    def tokenize(self, line: str) -> Tuple[List[T], List[int]]:
        tokens = self.text2tokens(line)
        ids = self.tokens2ids(tokens)
        return tokens, ids

    def detokenize(self, ids: List[int]) -> Tuple[str, List[T]]:
        tokens = self.ids2tokens(ids)
        text = self.tokens2text(tokens)
        return text, tokens

    @abstractmethod
    def text2tokens(self, line: str) -> List[T]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def tokens2text(self, tokens: List[T]) -> str:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def tokens2ids(self, tokens: List[T]) -> List[int]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def ids2tokens(self, ids: List[int]) -> List[T]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError("abstract method")

    @abstractproperty
    def symbol_table(self) -> Dict[T, int]:
        raise NotImplementedError("abstract method")


class HuggingFaceTokenizer(BaseTokenizer):

    def __init__(self, model: Union[str, PathLike], *args, **kwargs) -> None:
        # NOTE(Mddct): don't build here, pickle issues
        self.model = model
        self.tokenizer = None

        self.args = args
        self.kwargs = kwargs

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        recovery = {'tokenizer': None}
        self.__dict__.update(recovery)

    def _build_hugging_face(self):
        from transformers import AutoTokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, **self.kwargs)
            self.t2i = self.tokenizer.get_vocab()

    def text2tokens(self, line: str) -> List[Type]:
        self._build_hugging_face()
        return self.tokenizer.tokenize(line)

    def tokens2text(self, tokens: List[Type]) -> str:
        self._build_hugging_face()
        ids = self.tokens2ids(tokens)
        return self.tokenizer.decode(ids)

    def tokens2ids(self, tokens: List[Type]) -> List[int]:
        self._build_hugging_face()
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def ids2tokens(self, ids: List[int]) -> List[Type]:
        self._build_hugging_face()
        return self.tokenizer.convert_ids_to_tokens(ids)

    def vocab_size(self) -> int:
        self._build_hugging_face()
        # TODO: we need special tokenize size in future
        return len(self.tokenizer)

    @property
    def symbol_table(self) -> Dict[Type, int]:
        self._build_hugging_face()
        return self.t2i
