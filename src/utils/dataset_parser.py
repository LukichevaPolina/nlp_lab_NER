import pandas as pd


def parse_file(file_name: str) -> list:
    with open(file_name, 'r') as file:
        lines = [line.rstrip() for line in file][:-1]  # last string is empty

    data, sentence, tags = [], [], []

    for line in lines:
        if len(line) == 0:
            data.append([sentence, tags])
            sentence, tags = [], []
        else:
            current_line = line.split()
            sentence.append(current_line[1])
            tags.append(current_line[2])

    return data


def parse_dataset(file_name: str, columns=["Sentence", "Tags"]) -> pd.DataFrame:
    data_list = parse_file(file_name)
    df = pd.DataFrame(data_list, columns=columns)

    return df


def get_entities(tags_list):
    entities_list = []
    for tags in tags_list:
        entities = []
        start, end, entity = None, None, None
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if entity:
                    entities.append((start, end, entity))
                start, end, entity = i, i + 1, tag[2:]
            elif tag.startswith("I-") and entity == tag[2:]:
                end = i + 1
            else:
                if entity:
                    entities.append((start, end, entity))
                    start, end, entity = None, None, None
        if entity:
            entities.append((start, end, entity))
            print(entities)
        entities_list.append(entities)
    return entities_list


if __name__ == "__main__":
    parse_file("dataset/train.conllu")
