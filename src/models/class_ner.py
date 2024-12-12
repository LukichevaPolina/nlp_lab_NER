from torch import nn, Tensor

from typing import Any

# TODO: add word2vec encoder
# TODO: add initialization
class ClassNer(nn.Module):
    def __init__(self, encoder: Any, in_features: int, out_features: int):
        super().__init__()
        self._encoder = encoder
        self._classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features, out_features),
        )
        self._in_features = in_features
        self._out_features = out_features
    
    def forward(self, X: Tensor) -> Tensor:
        # encode first
        # classifier second
        pass 

    def encode(self, X: Tensor) -> Tensor:
        pass # It is not containt gradient here


