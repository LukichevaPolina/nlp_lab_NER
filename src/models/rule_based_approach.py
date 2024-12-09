import os
from pathlib import Path

from tqdm import tqdm
import spacy
from spacy.tokens import Doc
from spacy.pipeline import EntityRuler


from src.preprocessing.preprocessing import prepare_spacy_data

# TODO: implelent rule based approach
class Rulse_based_model:
    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    
    def _postprocess(self, doc: Doc) -> list:
        tag_dict = {"GPE" : "LOC",
                    "LOC" : "LOC",
                    "PERSON" : "PER",
                    "PER" : "PER",
                    "ORG" : "ORG",
                    "MISC" : "MISC",
                    "LANGUAGE" : "MISC",
                    "WORK_OF_ART" : "MISC"}

        res_list = []
        for token in doc:
            token_name = token.ent_type_
            if token_name in tag_dict.keys():
                if tag_dict[token_name] == "O":
                    res_list.append("O")
                else:
                    res_list.append( token.ent_iob_ + "-" + tag_dict[token_name])
            else:
                res_list.append("O")
        return res_list
    

    def fit(self, X, y):
        print("fit")
        ruler = self._nlp.add_pipe("entity_ruler")
        locations = set()
        organizations = set()
        misc_entities = set()
        for sentance, tags in zip(X, y):
            for idx, tag in enumerate(tags):
                if tag == "B-LOC" or tag == "I-LOC":
                    locations.add(sentance[idx])
                elif tag == "B-ORG" or tag == "I-ORG":
                    organizations.add(sentance[idx])
                elif tag == "B-MISC" or tag == "I-MISC":
                    misc_entities.add(sentance[idx])

        patterns = [{"label": "LOC", "pattern": [{"LOWER": loc}]} for loc in locations]
        patterns += [{"label": "ORG", "pattern": [{"LOWER": org}]} for org in organizations]
        patterns += [{"label": "MISC", "pattern": [{"LOWER": misc}]} for misc in misc_entities]
        print("add patterns")
        ruler.add_patterns(patterns)


    def predict(self, X) -> list:      
        print("predict")
        res_list = []
        for idx, sentence in enumerate(X['Sentence']):
            print(idx)
            doc = self._nlp(Doc(self._nlp.vocab, words=sentence))
            res_list.append(self._postprocess(doc))

        return res_list

