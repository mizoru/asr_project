import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    bs = len(dataset_items)
    n_features = dataset_items[0]["spectrogram"].size(1)
    result_batch = {}
    
    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["duration"] = [item["duration"] for item in dataset_items]

    # TODO: Maybe write a single loop for these?
    audio_length = [item["audio"].size(1) for item in dataset_items]
    max_length = max(audio_length)
    audio_batch = torch.zeros(bs, max_length)
    for i, item in enumerate(dataset_items):
        audio = item["audio"]
        audio_batch[i, :audio.size(1)] = audio
    result_batch["audio"] = audio_batch
    result_batch["audio_length"] = torch.tensor(audio_length)

    spectrogram_length = [item["spectrogram"].size(2) for item in dataset_items]
    max_length = max(spectrogram_length)
    spec_batch = torch.zeros(bs, n_features, max_length)
    for i, item in enumerate(dataset_items):
        spec = item["spectrogram"]
        spec_batch[i, :, :spec.size(2)] = spec
    result_batch["spectrogram"] = spec_batch
    result_batch["spectrogram_length"] = torch.tensor(spectrogram_length)

    text_encoded_length = [item["audio"].size(1) for item in dataset_items]
    max_length = max(text_encoded_length)
    text_batch = torch.zeros(bs, max_length)
    for i, item in enumerate(dataset_items):
        text = item["text_encoded"]
        text_batch[i, :text.size(1)] = text
    result_batch["text_encoded"] = text_batch
    result_batch["text_encoded_length"] = torch.tensor(text_encoded_length)

    return result_batch