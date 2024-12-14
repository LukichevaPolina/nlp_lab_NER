from torch import Tensor, nn
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import numpy as np
import torch
import pandas as pd
from src.models.class_ner.embeddings import IdfEmbedder, LabelEmbedder
from typing import Dict

from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(
        self, 
        sentences: pd.Series, 
        tags_list: pd.Series, 
        words2idf: Dict,
        tags_embedder: LabelEmbedder,
        max_sentence_len: int = 16,
    ) -> None:
        super().__init__()
        self._new_dataset = get_fix_sentence_len_dataset(sentences, tags_list, max_sentence_len)
        self._tags_embedder = tags_embedder
        self._word2idf = words2idf
    
    def __len__(self):
        return len(self._new_dataset)
    
    def __getitem__(self, index):
        sentence, tags, padding_slice = self._new_dataset["Sentence"][index], self._new_dataset["Tags"][index], self._new_dataset["Slice"][index]
        words_embedding = []
        for word in sentence:
            try:
                words_embedding.append(self._word2idf[word])
            except:
                words_embedding.append(self._word2idf["<unk>"])
            
        tags_embedding = self._tags_embedder.process(tags)

        return torch.as_tensor(words_embedding).float(), torch.as_tensor(tags_embedding).long(), torch.as_tensor(padding_slice).long()
    
class CustomDatamodule:
    def __init__(
        self, 
        sentences_train, tags_train, 
        sentences_test, tags_test, 
        words_embedder: IdfEmbedder,
        tags_embedder: LabelEmbedder,
        train_bs=64, test_bs=64, 
    ) -> None:
        self._sentences_train = sentences_train
        self._tags_train = tags_train
        self._sentences_test = sentences_test
        self._tags_test = tags_test
        self._train_bs = train_bs
        self._test_bs = test_bs

        self._tags_embedder = tags_embedder
        self._word2idf = words_embedder.process()
    
    def setup(self):
        self._train_dataset = CustomDataset(self._sentences_train, self._tags_train, self._word2idf, self._tags_embedder)
        self._test_dataset = CustomDataset(self._sentences_test, self._tags_test, self._word2idf, self._tags_embedder)

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._train_bs, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self._test_bs, shuffle=False)
    
def get_fix_sentence_len_dataset(sentences: pd.Series , tags_list: pd.Series, max_sentence_len: int = 16) -> pd.DataFrame:
    sentences_fix_length = []
    tags_fix_length = []
    padding_slice = []
    for idx, sentence in enumerate(sentences):
        start = 0
        tags = tags_list[idx]
        sentence_len = len(sentence)
        steps = int(sentence_len / max_sentence_len)
        for _ in range(steps):
            sentences_fix_length.append(sentence[start:start + max_sentence_len])
            tags_fix_length.append(tags[start:start + max_sentence_len])
            padding_slice.append(0)
            start += max_sentence_len

        diff = sentence_len - steps * max_sentence_len
        if diff > 0:
            dummy = sentence[start:sentence_len] + ["<unk>"] * (max_sentence_len - diff)
            dummy_tags = tags[start:sentence_len] + ["O"] * (max_sentence_len - diff)
            padding_slice.append(max_sentence_len - diff)
            sentences_fix_length.append(dummy)
            tags_fix_length.append(dummy_tags)
        
    return pd.DataFrame({"Sentence": sentences_fix_length, "Tags": tags_fix_length, "Slice": padding_slice})