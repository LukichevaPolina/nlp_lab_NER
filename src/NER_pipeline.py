from enum import Enum

import pandas as pd

from seqeval.metrics import classification_report, f1_score

from src.utils.dataset_parser import parse_dataset

from src.EDA.EDA import create_plots

from src.preprocessing.preprocessing import remove_punctuation, remove_whitespaces, lemamtization

from src.models.rule_based_approach import Rulse_based_model

from src.utils.dataset_parser import get_entities

class Preprocessor(Enum):
    NONE = 1
    REMOVE_PUNCTUATION = 2

class Algorithm(Enum):
    RULE_BASED = 1
    NN = 2

class Mode(Enum):
    TRAIN = 1
    EVAL = 2
    INFER = 3

TARGET2ENUM = {
    "rule-based": Algorithm.RULE_BASED,
    "nn": Algorithm.NN,
    "prepoc-none": Preprocessor.NONE,
    "remove-punctuation": Preprocessor.REMOVE_PUNCTUATION,
    "remove-whitespaces": Preprocessor.REMOVE_WHITESPACES,
    "train": Mode.TRAIN,
    "eval": Mode.EVAL,
    "infer": Mode.INFER
}

ENUM2TARGET = dict(zip(TARGET2ENUM.values(), TARGET2ENUM.keys()))

class NER_pipeline:
    def __init__(self,
                 dataset_folder: str,
                 train_dataset_name: str,
                 test_dataset_name: str,
                 val_dataset_name: str,
                 algorithm: str,
                 preprocessor: str,
                 embedder: str,
                 mode: str,
                 ) -> None:
        self._train_dataset = parse_dataset(
            dataset_folder + train_dataset_name)
        self._test_dataset = parse_dataset(dataset_folder + test_dataset_name)
        self._val_dataset = parse_dataset(dataset_folder + val_dataset_name)
        self._algorithm = self.str2enum(algorithm)
        self._preprocessor = self.str2enum(preprocessor)
        self._embedder = self.str2enum(embedder)
        self._mode = self.str2enum(mode)

        self.model = None

    def run(self) -> None:
        self.preprocess()
        self.train()
        self.eval()

    def preprocess(self) -> None:
        # crete plots
        create_plots([self._train_dataset, self._test_dataset,
                     self._val_dataset], ["train", "test", "val"])

        # preprocessing
        if self._preprocessor == Preprocessor.REMOVE_PUNCTUATION:
            self._train_dataset = remove_punctuation(self._train_dataset)
            self._test_dataset = remove_punctuation(self._test_dataset)
            self._val_dataset = remove_punctuation(self._val_dataset)

        if self._algorithm == Algorithm.NN:
            self._train_dataset = lemamtization(self._train_dataset)
            self._test_dataset = lemamtization(self._test_dataset)
            self._val_dataset = lemamtization(self._val_dataset)

    def train(self) -> None:
        if self._algorithm == Algorithm.RULE_BASED:
            self.model = Rulse_based_model(is_use_custom_rules=True)
            self.model.fit(
                self._train_dataset["Sentence"], self._train_dataset["Tags"])
        elif self._algorithm == Algorithm.NN:
            raise NotImplementedError(f"Algorithm doesn't supported yet")
        else:
            raise RuntimeError(f"Algorithm: {self._algorithm} is not supported")

    def eval(self) -> None:
        if self._algorithm == Algorithm.RULE_BASED:
            res = self.model.predict(self._train_dataset["Sentence"])
            res_true = self._train_dataset["Tags"]

            print(classification_report(res, res_true))
            print(f"f1-score: {f1_score(res, res_true)}")
        elif self._algorithm == Algorithm.NN:
            raise NotImplementedError(f"Algorithm doesn't supported yet")
        else:
            raise RuntimeError(f"Algorithm: {self._algorithm} is not supported")

    def str2enum(self, target: str) -> Algorithm:
        try:
            return TARGET2ENUM[target]
        except:
            raise ValueError(f"Given algorithm: {target} does not exist")
