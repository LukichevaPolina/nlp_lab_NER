from enum import Enum

import pandas as pd

from seqeval.metrics import classification_report, f1_score

from src.utils.dataset_parser import parse_dataset

from src.EDA.EDA import create_plots

from src.preprocessing.preprocessing import remove_punctuation, lemamtization

from src.models.rule_based_approach import Rulse_based_model

from src.utils.dataset_parser import get_entities

from src.models.class_ner.train import train

from src.utils.rendering import (
    plot_accuracy_curve, plot_f1_curve, plot_learning_curve
)

class Preprocessor(Enum):
    NONE = 1
    REMOVE_PUNCTUATION = 2

class Algorithm(Enum):
    RULE_BASED = 1
    NN = 2

class Mode(Enum):
    TRAIN = 1
    EVAL = 2

TARGET2ENUM = {
    "rule-based": Algorithm.RULE_BASED,
    "nn": Algorithm.NN,
    "prepoc-none": Preprocessor.NONE,
    "remove-punctuation": Preprocessor.REMOVE_PUNCTUATION,
    "train": Mode.TRAIN,
    "eval": Mode.EVAL,
}

ENUM2TARGET = dict(zip(TARGET2ENUM.values(), TARGET2ENUM.keys()))

class NER_pipeline:
    def __init__(self,
                 dataset_folder: str,
                 train_dataset_name: str,
                 test_dataset_name: str,
                 algorithm: str,
                 preprocessor: str,
                 mode: str,
        ) -> None:
        self._train_dataset = parse_dataset(
            dataset_folder + train_dataset_name)
        self._test_dataset = parse_dataset(dataset_folder + test_dataset_name)
        self._algorithm = self.str2enum(algorithm)
        self._preprocessor = self.str2enum(preprocessor)
        self._mode = self.str2enum(mode)

        self.model = None
        self.dl_model = None

    def run(self) -> None:
        self.preprocess()
        if self._mode == Mode.TRAIN:
            self.train()
        elif self._mode == Mode.EVAL:
            self.eval()

    def preprocess(self) -> None:
        # crete plots
        create_plots([self._train_dataset, self._test_dataset], ["train", "test"])

        # preprocessing
        if self._preprocessor == Preprocessor.REMOVE_PUNCTUATION:
            self._train_dataset = remove_punctuation(self._train_dataset)
            self._test_dataset = remove_punctuation(self._test_dataset)

        if self._algorithm == Algorithm.NN:
            self._train_dataset = lemamtization(self._train_dataset)
            self._test_dataset = lemamtization(self._test_dataset)

    def train(self) -> None:
        if self._algorithm == Algorithm.RULE_BASED:
            self.model = Rulse_based_model(is_use_custom_rules=True)
            self.model.fit(
                self._train_dataset["Sentence"], self._train_dataset["Tags"])
        elif self._algorithm == Algorithm.NN:
            train_metrics, val_metrics, train_losses, val_losses = train(chekpoint_save="checkpoints/class_ner.pt", train_dataset=self._train_dataset, test_dataset=self._test_dataset)
            pd.DataFrame(train_metrics).to_csv(
                f"metrics/train_metrics.csv")
            pd.DataFrame(val_metrics).to_csv(
                f"metrics/val_metrics.csv")
            pd.DataFrame(train_losses).to_csv(
                f"metrics/train_losses.csv")
            pd.DataFrame(val_losses).to_csv(
                f"metrics/val_losses.csv")
            plot_learning_curve(
                train_losses["ce"], val_losses["ce"], name="cnn_learning_curve")
            plot_f1_curve(
                train_metrics["f1"], val_metrics["f1"], name="cnn_f1_curve")
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
