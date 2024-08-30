from collections import defaultdict
from typing import Any
from nltk import ngrams
from ggd_py_utils.formating.numeric import abbreviate_large_number
from ggd_py_utils.tracing.metrics import time_block

def get_words_and_subwords_counts(filename:str, ngram_range:tuple=(2, 6)) -> dict:
    text:str = ""

    with time_block(block_name="Read File"):
        with open(file=filename, mode='r', encoding="utf-8") as file:
            text = file.read()

    words:list[str]
    unique_words:set
    word_count:int
    
    with time_block(block_name="Words Count"):
        words = text.split()
        unique_words = set(words)
        word_count = len(unique_words)

    ngram_counts = defaultdict(int)
    subwords_ngrams = 0

    min_n, max_n = ngram_range

    normalized_text: str = ' '.join(text.split())

    with time_block(block_name="Subwords Count"):
        for n in range(min_n, max_n + 1):
            _ngrams: zip[Any] = ngrams(normalized_text, n)
            
            for ngram in _ngrams:
                ngram_counts[''.join(ngram)] += 1
                subwords_ngrams += 1

    counts: dict[str, int] = {
        "words": word_count,
        "words_abbreviated": abbreviate_large_number(number=word_count),
        "subwords": subwords_ngrams,
        "subwords_abbreviated": abbreviate_large_number(number=subwords_ngrams),
    }

    return counts
