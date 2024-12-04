from string import punctuation

import pandas as pd


# TODO: improve this dummy implementation
def remove_punctuation(data: pd.DataFrame) -> pd.DataFrame:
    punctuations_str = punctuation
    for row_idx, row in enumerate(data.itertuples()):
        sentence = []
        tags = []
        for idx, token in enumerate(row.Sentence):
            if token not in punctuations_str:
                sentence.append(token)
                tags.append(row.Tags[idx])
        data.loc[row_idx, "Sentence"] = sentence
        data.loc[row_idx, "Tags"] = tags

    return data


# TODO: implement embedders
def get_embedings(data: pd.DataFrame):
    pass
