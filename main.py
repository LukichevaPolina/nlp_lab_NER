import sys
import argparse
from typing import Dict

from src.NER_pipeline import NER_pipeline


def parse_args() -> Dict[str, str]:
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_folder",
        required=False,
        default="dataset/",
        type=str,
        help="Path to the dataset folder"
    )
    args.add_argument(
        "--train_dataset_name",
        required=False,
        default="train.conllu",
        type=str,
        help="Name of file with train dataset"
    )
    args.add_argument(
        "--test_dataset_name",
        required=False,
        default="test.conllu",
        type=str,
        help="Name of file with test dataset"
    )
    args.add_argument(
        "--algorithm",
        required=True,
        type=str,
        choices=["rule-based", "nn"],
        help="algorithm to process"
    )
    args.add_argument(
        "--preprocessor",
        type=str,
        default="prepoc-none",
        # TODO: replace-capital-letters - ?
        choices=["prepoc-none", "remove-punctuation",
                 "remove-all"],
        help="the strategy to do preprocess"
    )
    args.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["train", "eval"],
        help="mode which used: \
              train - model will be trained and evaluated, \
              eval  - model will be evaluated"
    )

    return vars(args.parse_args())


def main() -> None:
    args = parse_args()
    pipeline = NER_pipeline(
        dataset_folder=args["dataset_folder"],
        train_dataset_name=args["train_dataset_name"],
        test_dataset_name=args["test_dataset_name"],
        algorithm=args["algorithm"],
        preprocessor=args["preprocessor"],
        mode=args["mode"]
    )

    pipeline.run()


if __name__ == "__main__":
    sys.exit(main() or 0)
