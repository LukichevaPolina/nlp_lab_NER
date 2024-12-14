import pandas as pd
from typing import List
import numpy as np

def create_vocabulary(sentences: pd.Series, vocabulary_num=20000) -> List:
    unique_words, counts = np.unique(np.array([i for data_list in sentences for i in data_list]), return_counts=True)
    frequency_map = {key: value for key, value in zip(unique_words, counts)}
    top_frequency_map = {key: value for key, value in sorted(frequency_map.items(), key=lambda item: item[1], reverse=True)}

    vocabulary = list(top_frequency_map.keys())[:vocabulary_num]

    return list(map(lambda word: str(word), vocabulary))