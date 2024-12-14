from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

class IdfEmbedder:
    def __init__(self, vocabulary, token_pattern='(?u)\\b\\w\\w+\\b', use_idf=True):
        self._vectorizer = TfidfVectorizer(vocabulary=vocabulary, token_pattern=token_pattern, use_idf=use_idf)

    def train(self, corpus, save_ckpt="checkpoints/tfidf.pkl"):
        self._vectorizer.fit_transform(corpus)
        joblib.dump(self._vectorizer, save_ckpt)

    def process(self, ckpt="checkpoints/tfidf.pkl"):
        if ckpt is not None:
            self._vectorizer = joblib.load(ckpt)
        else:
            raise RuntimeError("Vectorizer is not exist")

        word2idf = dict(zip(self._vectorizer.get_feature_names_out(), self._vectorizer.idf_))
        word2idf["<unk>"] = np.float64(0)

        return word2idf

class LabelEmbedder:
    def __init__(self):
        self._target_encoder = LabelEncoder()
        self._tag_list = ['O', 'B-LOC',  'I-LOC', 'B-ORG',  'I-ORG',  'B-MISC',  'I-MISC', 'B-PER', 'I-PER']
        self._target_encoder.fit(self._tag_list)

    def process(self, tag_list):
        return self._target_encoder.transform(tag_list)

    def inverse(self, tag_list):
        return self._target_encoder.inverse_transform(tag_list)