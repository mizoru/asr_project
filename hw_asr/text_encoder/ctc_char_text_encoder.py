from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder
from hw_asr.text_encoder.lm_decoder import setup_lm_decoder


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
        
        probs = probs.detach().cpu().numpy()
        probs_length = probs_length.detach().item()
        

        def extend_and_merge_beam(hypos, prob, last=False):
            new_hypos = defaultdict(float)
            for char_idx, char_prob in enumerate(prob):
                new_char = self.ind2char[char_idx]
                for hypo in hypos:
                    text = hypo.text
                    if len(text) == 0 or text[-1] != new_char:
                        text = text.strip(self.EMPTY_TOK) + new_char
                    if last: text = text.strip(self.EMPTY_TOK)
                    new_hypos[text] += char_prob * hypo.prob
            new_hypos = [Hypothesis(text, prob) for text, prob in new_hypos.items()]
            return new_hypos

        def truncate_beam(hypos, beam_size):
            return sorted(hypos, key=lambda x: x.prob, reverse=True)[:beam_size]

        hypos: List[Hypothesis] = [Hypothesis("", 1.)]
        for idx in range(probs_length):
            hypos = extend_and_merge_beam(
                hypos, probs[idx], last=idx == probs_length-1)
            hypos = truncate_beam(hypos, beam_size)
        return hypos
    
    def ctc_decode_lm(self, probs: torch.tensor, probs_length):
        if not self.decoder:
            vocab = [""] + list(self.alphabet)
            self.decoder = setup_lm_decoder(vocab)
        return self.decoder.decode(probs[:probs_length-1])
        
