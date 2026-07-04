import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import List
import click
import emoji
import pandas as pd
from datasets import DatasetDict, load_dataset, Dataset, concatenate_datasets
import json
from pathlib import Path
from typing import Callable, Tuple, Dict, Union, Any

import numpy as np
import seaborn
import torch
from datasets import load_dataset, load_from_disk, ClassLabel
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoTokenizer

DATASET_SPLITS = ["train", "validation", "test"]
TOKENIZE_COLUMNS = ["input_ids", "attention_mask", "labels"]

dataset_to_input_output = {
    "davidson": {
        "input": "tweet",
        "output": "class",
    },
    "talat_hovy": {
        "input": "text",
        "output": "label",
    },
    "vidgen": {
        "input": "text",
        "output": "label",
    },
    "mathew": {
        "input": "sentence",
        "output": "label",
    },
    "kennedy": {
        "input": "text",
        "output": "label",
    },
    "founta": {
        "input": "Tweet text",
        "output": "Label",
    },
    "Paul/hatecheck": {
            "input": "test_case",
            "output": "label_gold",
        }
}





DATASETS = ["davidson"] #, "founta", "kennedy", "mathew", "talat_hovy", "vidgen"]


class HFDatasetCreator:
    """
        TBD.
    """
    def __init__(self, dataset_name: str, dataset_file: str, dataset_split: List[float], seed: int = 0):
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        if dataset_split:
            self.train_split = dataset_split[0]
            self.val_split = dataset_split[1]
            self.test_split = dataset_split[2]
            assert self.train_split + self.val_split + self.test_split == 1, "The split distribution does not sum up " \
                                                                             "to 1."
        self.seed = seed
        self.dataset = None

    def load_data_from_file(self):
        if self.dataset_name == "talat_hovy":
            file_names_and_labels = {
                "neither.json": "neither",
                "racism.json": "racism",
                "sexism.json": "sexism"
            }

            datasets = []
            for file_name, label in file_names_and_labels.items():
                path = self.dataset_file / Path(file_name)
                examples = [json.loads(line) for line in open(path, "r")]
                [example.update({"label": label}) for example in examples]
                dataset = Dataset.from_list(examples)
                remove_columns = dataset.features.keys() - ["text", "label"]
                dataset = dataset.remove_columns(remove_columns)
                datasets.append(dataset)
            dataset = concatenate_datasets(datasets)
            self.dataset = DatasetDict({"train": dataset})
        elif self.dataset_name == "founta":
            dataset = pd.read_csv(self.dataset_file, delimiter=r"\s{2,}|\t")
            dataset = Dataset.from_pandas(dataset)
            self.dataset = DatasetDict({"train": dataset})
        else:
            suffix = Path(self.dataset_file).suffix[1:]
            self.dataset = load_dataset(suffix, data_files=self.dataset_file)

    def load_data_from_name(self):
        self.dataset = load_dataset(self.dataset_file)

    def split_dataset(self):
        if self.dataset_name in ["davidson", "founta", "kennedy", "talat_hovy"]:
            test_val_split = self.val_split + self.test_split
            train_testvalid = self.dataset["train"].train_test_split(test_size=test_val_split, seed=self.seed)

            left_over_split = 1 - self.train_split
            new_test_split = self.test_split / left_over_split

            valid_test = train_testvalid["test"].train_test_split(test_size=new_test_split, seed=self.seed)

            self.dataset = DatasetDict({
                "train": train_testvalid["train"],
                "val": valid_test["train"],
                "test": valid_test["test"],
            })
        elif self.dataset_name == "vidgen":
            train_data = self.dataset.filter(lambda example: example["split"] == "train")
            val_data = self.dataset.filter(lambda example: example["split"] == "dev")
            test_data = self.dataset.filter(lambda example: example["split"] == "test")

            self.dataset = DatasetDict({
                "train": train_data["train"],
                "val": val_data["train"],
                "test": test_data["train"],
            })

    def save_dataset(self, output: str):
        self.dataset.save_to_disk(output)

    def prepare_kennedy(self):
        def convert_score_label(hate_speech_score: float):
            label = "nothate"
            if hate_speech_score > 0.5:
                label = "hate"
            return label

        # Get labels from the scores based on average of the annotators.
        dataset = pd.DataFrame(self.dataset["train"])
        comment_ids = list(set(dataset["comment_id"]))

        labels = {}
        for comment_id in comment_ids:
            score = dataset[dataset["comment_id"] == comment_id]["hate_speech_score"].mean()
            label = convert_score_label(score)
            labels[comment_id] = label

        # Assign final label to the comments and remove duplicates.
        remove_columns = self.dataset["train"].features.keys() - ["comment_id", "text"]
        dataset = self.dataset["train"].remove_columns(remove_columns)
        dataset = pd.DataFrame(dataset).drop_duplicates()

        dataset["label"] = dataset.apply(lambda x: labels[x["comment_id"]], axis=1)
        self.dataset["train"] = Dataset.from_pandas(dataset)

    def prepare_mathew(self):
        def get_most_common_label(example):
            labels = example["annotators"]["label"]
            most_common_label = Counter(labels).most_common(1)[0][0]
            example["label"] = most_common_label
            return example

        def tokens_to_sentence(example):
            example["sentence"] = " ".join(example["post_tokens"])
            return example

        dataset = self.dataset.map(get_most_common_label)
        self.dataset = dataset.map(tokens_to_sentence)

    def clean_data(self, emojis: bool = True, urls: bool = True, usernames: bool = True):
        def remove_emojis(example):
            input_key = dataset_to_input_output[self.dataset_name]["input"]
            example[input_key] = emoji.demojize(example[input_key])
            return example

        def remove_urls(example):
            input_key = dataset_to_input_output[self.dataset_name]["input"]

            url_pattern = re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[^\s]+"
            )
            example[input_key] = re.sub(url_pattern, "[URL]", example[input_key])

            return example

        def remove_usernames(example):
            input_key = dataset_to_input_output[self.dataset_name]["input"]

            username_pattern = re.compile(r'@[A-Za-z0-9_]+')
            example[input_key] = re.sub(username_pattern, "[USER]", example[input_key])
            return example

        if emojis:
            self.dataset = self.dataset.map(remove_emojis)
        if urls:
            self.dataset = self.dataset.map(remove_urls)
        if usernames:
            self.dataset = self.dataset.map(remove_usernames)


@click.command()
@click.option("-n", "--dataset-name", "dataset_name", required=True, type=str)
@click.option("-p", "--path", "path", type=str)
@click.option("-o", "--output", "output", required=True, type=str)
@click.option("-s", "--dataset-split", "dataset_split", type=str)
def main(dataset_name: str, path: str, output: str, dataset_split: str):
    assert dataset_name in DATASETS, f"This dataset is not supported, please provide one of the following datasets: " \
                                     f"{DATASETS} "

    if dataset_split:
        dataset_split = ast.literal_eval(dataset_split)
    if dataset_name == "davidson":
        davidson_creator = HFDatasetCreator(dataset_name, path, dataset_split)
        davidson_creator.load_data_from_file()
        davidson_creator.clean_data()
        davidson_creator.split_dataset()
        davidson_creator.save_dataset(output)
    elif dataset_name == "founta":
        founta_creator = HFDatasetCreator(dataset_name, path, dataset_split)
        founta_creator.load_data_from_file()
        founta_creator.clean_data()
        founta_creator.split_dataset()
        founta_creator.save_dataset(output)
    elif dataset_name == "vidgen":
        vidgen_creator = HFDatasetCreator(dataset_name, path, dataset_split)
        vidgen_creator.load_data_from_file()
        vidgen_creator.clean_data()
        vidgen_creator.split_dataset()
        vidgen_creator.save_dataset(output)
    elif dataset_name == "kennedy":
        kennedy_creator = HFDatasetCreator(dataset_name, "ucberkeley-dlab/measuring-hate-speech", dataset_split)
        kennedy_creator.load_data_from_name()
        kennedy_creator.prepare_kennedy()
        kennedy_creator.clean_data()
        kennedy_creator.split_dataset()
        kennedy_creator.save_dataset(output)
    elif dataset_name == "mathew":
        mathew_creator = HFDatasetCreator(dataset_name, "hatexplain", dataset_split)
        mathew_creator.load_data_from_name()
        mathew_creator.prepare_mathew()
        mathew_creator.clean_data()
        mathew_creator.save_dataset(output)
    elif dataset_name == "talat_hovy":
        talat_hovy_creator = HFDatasetCreator(dataset_name, path, dataset_split)
        talat_hovy_creator.load_data_from_file()
        talat_hovy_creator.clean_data()
        talat_hovy_creator.split_dataset()
        talat_hovy_creator.save_dataset(output)


if __name__ == "__main__":
    main()
