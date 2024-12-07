from string import punctuation

import pandas as pd


# TODO: fix bug
def remove_punctuation(data: pd.DataFrame) -> pd.DataFrame:
    punctuations_str = punctuation
    for row_idx, row in enumerate(data.itertuples()):
        sentence = ""
        tags = []
        for idx, token in enumerate(row.Sentence.split()):
            if token not in punctuations_str:
                sentence += token + " "
                tags.append(row.Tags[idx])

        data.loc[row_idx, "Sentence"] = sentence
        data.loc[row_idx, "Tags"] = tags

    return data


# TODO: implement embedders
def get_embedings(data: pd.DataFrame):
    pass


def prepare_spacy_data(df: pd.date_range) -> list:
    spacy_data = []
    for sentance, tags in zip(df['Sentence'], df['Tags']):
        entities = []
        start = 0
        for token, tag in zip(sentance.split(), tags):
            if tag != "O":  # Если это не фон
                tag_split = tag.split("-")
                label = tag_split[1]
                entities.append((start, start + len(token), label))
            start += len(token) + 1  # Учитываем пробел
        spacy_data.append((" ".join(token), {"entities": entities}))
    return spacy_data
