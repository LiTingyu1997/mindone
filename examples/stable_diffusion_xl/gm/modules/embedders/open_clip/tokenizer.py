# This code is adapted from https://github.com/mlfoundations/open_clip
# with modifications to run on MindSpore.

""" CLIP tokenizer

Reference to https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import gzip
import html
import os
from functools import lru_cache
from typing import List, Union

import ftfy
import numpy as np
import regex as re

from mindspore import Tensor

# https://stackoverflow.com/q/62691279
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe(), special_tokens=None):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        if not special_tokens:
            special_tokens = ["<start_of_text>", "<end_of_text>"]
        else:
            special_tokens = ["<start_of_text>", "<end_of_text>"] + special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE
        )

        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace").replace("</w>", " ")
        return text


_tokenizer = SimpleTokenizer()


def decode(output_ids: Tensor):
    output_ids = output_ids.asnumpy()
    return _tokenizer.decode(output_ids)


def lpw_tokenize(
    texts: Union[str, List[str]],
    context_length: int = 77,
    max_embeddings_multiples: int = 4,
) -> [np.ndarray, np.ndarray]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]
    max_length = (context_length - 2) * max_embeddings_multiples + 2
    prompt_tokens = [
        _tokenizer.encode(text)
        if len(_tokenizer.encode(text)) <= max_length - 2
        else _tokenizer.encode(text)[: max_length - 2]
        for text in texts
    ]
    prompt_tokens_length = np.array([len(p) + 2 for p in prompt_tokens], np.int32)
    max_length = max([len(token) for token in prompt_tokens])

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (context_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (context_length - 2) * max_embeddings_multiples + 2
    # pad the length of tokens and weights
    bos = _tokenizer.encoder["<start_of_text>"]
    eos = _tokenizer.encoder["<end_of_text>"]
    pad = 0
    for i in range(len(prompt_tokens)):
        prompt_tokens[i] = [bos] + prompt_tokens[i] + [eos] + [pad] * (max_length - 1 - len(prompt_tokens[i]) - 1)
    prompt_tokens = np.array(prompt_tokens, np.int32)

    return prompt_tokens, prompt_tokens_length


def lpw_tokenize2(
    texts: Union[str, List[str]],
    context_length: int = 77,
    max_embeddings_multiples: int = 4,
) -> [np.ndarray, np.ndarray]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]
    max_length = (context_length - 2) * max_embeddings_multiples + 2
    prompt_tokens = [
        _tokenizer.encode(text)
        if len(_tokenizer.encode(text)) <= max_length - 2
        else _tokenizer.encode(text)[: max_length - 2]
        for text in texts
    ]
    prompt_tokens_length = np.array([len(p) + 2 for p in prompt_tokens], np.int32)

    # max_length = max([len(token) for token in prompt_tokens])
    # max_embeddings_multiples = min(
    #     max_embeddings_multiples,
    #     (max_length - 1) // (context_length - 2) + 1,
    # )
    # max_embeddings_multiples = max(1, max_embeddings_multiples)

    max_length = (context_length - 2) * max_embeddings_multiples + 2
    # pad the length of tokens and weights
    bos = _tokenizer.encoder["<start_of_text>"]
    eos = _tokenizer.encoder["<end_of_text>"]
    pad = 0
    for i in range(len(prompt_tokens)):
        prompt_tokens[i] = [bos] + prompt_tokens[i] + [pad] * (max_length - 1 - len(prompt_tokens[i]) - 1) + [eos]
    prompt_tokens = np.array(prompt_tokens, np.int32)

    return prompt_tokens, prompt_tokens_length


def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> [np.ndarray, np.ndarray]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, np.ndarray):  # MindData's `.map` returns strings wrapped into a numpy array
        texts = texts.item()
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<start_of_text>"]
    eot_token = _tokenizer.encoder["<end_of_text>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), np.int32)
    length = np.zeros(len(all_tokens), np.int32)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token
        result[i, : len(tokens)] = np.array(tokens, np.int32)
        length[i] = len(tokens)

    return result, length
