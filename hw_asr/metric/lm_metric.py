from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer


class LMDecoderWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []

        predictions = [self.ctc_decode_lm(
            log_probs[i], log_probs_length[i]) for i in range(log_probs.size(0))]

        for pred_text, target_text in zip(predictions, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred_text))
            
        return sum(wers) / len(wers)
