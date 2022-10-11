from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        chars = ['']
        for ind in inds:
            char = self.ind2char[ind]
            if char == self.EMPTY_TOK:
                char = ''
            if char != chars[-1]:
                chars.append(char)
        return ''.join(chars)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = [Hypothesis("", 1.)]

        def extend_and_merge_beam(hypos, idx):
            new_hypos: List[Hypothesis] = []
            for text, prob in hypos:
                for char_i in range(voc_size):
                    new_char = self.ind2char[char_i]
                    if len(text) == 0 or new_char != text[-1]:
                        text = text.strip(self.EMPTY_TOK) + new_char
                    new_hypos.append(Hypothesis(text, prob*probs[idx, char_i]))
            merged = defaultdict(float)
            for hypo in new_hypos:
                merged[hypo.text] += hypo.prob
            new_hypos = [Hypothesis(text, prob) for (text, prob) in merged.items()]
            return new_hypos

        def truncate_beam(hypos, beam_size):
            return sorted(hypos, key=lambda x: x.prob, reverse=True)[:beam_size]

        for idx in range(probs_length):
            hypos = extend_and_merge_beam(hypos, idx)
            hypos = truncate_beam(hypos, beam_size)
        return hypos
