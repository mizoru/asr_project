from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class RNNWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.RNN(*args, **kwargs)

    def forward(self, x):
        x, _ = self.net(x)
        return x


class DeepSpeech2(BaseModel):
    def __init__(
            self,
            n_feats=128,
            conv_channels=[32, 32, 96],
            time_stride=2,
            rnn_hidden=1024,
            n_class=28,
            n_rnn=6
    ):
        super().__init__(n_feats, n_class)
        self.cnn = Sequential(
            nn.Conv2d(1, conv_channels[0],
                      (41, 11), (2, time_stride), padding=(20, 5)),
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1],
                      (21, 11), (2, 1), padding=(0, 5)),
            nn.ReLU(),
            nn.Conv2d(conv_channels[1], conv_channels[2],
                      (21, 11), (2, 1), padding=(0, 5)),
            nn.ReLU()
        )
        rnn_layers = [RNNWrapper(conv_channels[2], rnn_hidden, bidirectional=True,
                                 batch_first=True, nonlinearity='relu'), nn.LazyBatchNorm1d()]
        for i in range(n_rnn-1):
            rnn_layers.append(RNNWrapper(rnn_hidden*2, rnn_hidden,
                              bidirectional=True, batch_first=True, nonlinearity='relu'))
            rnn_layers.append(nn.LazyBatchNorm1d())

        self.rnn = Sequential(*rnn_layers)
        self.linear = nn.Linear(rnn_hidden*2, n_class)

    def forward(self, spectrogram, **batch):
        x = spectrogram.unsqueeze(1)
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.transpose(1, 2)
        x = self.rnn(x)
        x = self.linear(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths - 1)//2 + 1
