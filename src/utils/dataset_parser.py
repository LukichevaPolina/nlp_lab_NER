import pandas as pd

def parse_file(file_name: str) -> list:
    with open(file_name, 'r') as file:
        lines = [line.rstrip() for line in file][:-1] # last string is empty
    
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


if __name__ == "__main__":
    parse_file("dataset/train.conllu")