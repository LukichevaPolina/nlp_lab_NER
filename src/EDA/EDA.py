import logging as log

import matplotlib.pyplot as plt
import pandas as pd


SAVE_PATH = "plots/"
TAGS_LIST = ['O', 'B-LOC',  'I-LOC', 'B-ORG',  'I-ORG',  'B-MISC',  'I-MISC', 'B-PER', 'I-PER']


def create_plots(data_list: list, dataset_type_list: list) -> None:
    for idx, dataset in enumerate(data_list):
        sentence_length_distrbution(dataset, dataset_type_list[idx])
        tags_distribution_plot(dataset, dataset_type_list[idx])
        tags_word_position_dependency(dataset, dataset_type_list[idx])


def get_tags_statistic(data: pd.DataFrame) -> dict:
    stat_dict = { tag: 0 for tag in TAGS_LIST}

    for row in data["Tags"]:
        for tag in row:
            stat_dict[tag] += 1

    return stat_dict


def get_tags_positions(data: pd.DataFrame) -> dict:
    stat_dict = { tag: [] for tag in TAGS_LIST}
    for row in data["Tags"]:
        for idx, tag in enumerate(row):
            stat_dict[tag].append(idx) 
    
    return stat_dict


# with_O add O distribuution to the plot
def tags_distribution_plot(data: pd.DataFrame, dataset_type: str, without_O=True) -> None:
    log.info("PLOT: save class distribution into graphs/")

    statistic = get_tags_statistic(data)

    if (without_O):
        statistic.pop("O")

    fig = plt.figure(figsize=(9, 5))
    plt.bar(statistic.keys(), statistic.values())
    fig.suptitle(f"Tags distribution for {dataset_type} dataset", fontsize=15, fontweight='bold')
    plt.xlabel("tag name", fontsize=9, fontweight='bold')
    plt.ylabel("frequency", fontsize=9, fontweight='bold')
    fig.savefig(f"{SAVE_PATH}tags_distribution_{dataset_type}_O_tag_{not without_O}.png")


def tags_word_position_dependency(data: pd.DataFrame, dataset_type: str) -> None:
    statistic = get_tags_positions(data)
    statistic.pop("O")

    fig, ax = plt.subplots(nrows=2, ncols=4, sharex=False, sharey=True, figsize=(17, 10))
    fig.suptitle(f"Tag/word position distrubution for {dataset_type} dataset", fontsize=15, fontweight='bold')

    tags = TAGS_LIST[1:] # remove "O" tag

    for idx, tag_name in enumerate(tags):
        ax[idx // 4, idx % 4].hist(statistic[tag_name], bins=30, color="g")
        ax[idx // 4, idx % 4].title.set_text(f"{tag_name}")
        ax[idx // 4, idx % 4].set(xlabel=f"word position", ylabel="frequency")

    fig.savefig(f"{SAVE_PATH}tag_word_position_distribution_{dataset_type}.png")
    

def sentence_length_distrbution(data: pd.DataFrame, dataset_type: str) -> None:
    sentences_lentghts = [len(sentence) for sentence in data["Sentence"]]
    fig = plt.figure(figsize=(9, 5))
    plt.hist(sentences_lentghts, bins=50, histtype="barstacked")
    fig.suptitle(f"Sentences length distribution in {dataset_type} dataset", fontsize=15, fontweight='bold')
    plt.xlabel("sentence length", fontsize=9, fontweight='bold')
    plt.ylabel("frequency", fontsize=9, fontweight='bold')
    fig.savefig(f"{SAVE_PATH}sentence_length_distribution_{dataset_type}.png")
