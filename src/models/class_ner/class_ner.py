from torch import nn, Tensor
import torch

from typing import List

from typing import Any

class ClassNer(nn.Module):
    def __init__(self, in_features: int = 16, out_features: int = 9, hidden_features: int = 500, num_heads: int = 16):
        super().__init__()
        self._feed_forward = nn.Sequential(
            nn.Linear(in_features, int(in_features/2)),
            nn.ReLU(),
            nn.Linear(int(in_features/2), in_features)
        )
        self._embedders = nn.ModuleDict({
            f"embedder_{i}":
            nn.Sequential(
                nn.Linear(1, hidden_features),
                nn.ReLU(),
            ) 
            for i in range(num_heads)
        })
        self._classifiers = nn.ModuleDict({
            f"classifier_{i}":
            nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, out_features)
            )
            for i in range(num_heads)
        })
    
    def forward(self, x: Tensor) -> List[Tensor]:
        #print(f"{x.shape=}")

        feed_forward_out: Tensor = self._feed_forward(x)
        #print(f"{feed_forward_out.shape=}")
        # reshape to (64, 1, 1, 16)

        words = torch.split(feed_forward_out, 1, dim=3)

        words_embeddings = []
        for i, word in enumerate(words):
            words_embeddings.append(self._embedders[f"embedder_{i}"](word))
        
        #print(f"{words_embeddings[0].shape=}")

        logits = []
        for i, embedding in enumerate(words_embeddings):
            logits.append(self._classifiers[f"classifier_{i}"](embedding))
        #print(f"{logits[0].shape=}")

        return logits
    
    @staticmethod
    def initialize(net) -> None:
        if isinstance(net, nn.Linear):
            if net.weight is not None:
                nn.init.xavier_uniform_(net.weight.data, gain=1)
            if net.bias is not None:
                nn.init.constant_(net.bias.data, 0.0)
         

    


