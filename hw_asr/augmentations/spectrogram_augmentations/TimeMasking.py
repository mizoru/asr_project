import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class TimeMasking(AugmentationBase):
    def __init__(self, time_mask_param=10, p=0.1):
        self._aug = RandomApply(torchaudio.transforms.TimeMasking(time_mask_param), p=p)

    def __call__(self, data: Tensor):
        return self._aug(data)