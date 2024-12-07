import os
from pathlib import Path

from tqdm import tqdm
import spacy
from spacy.tokens import Doc

from src.preprocessing.preprocessing import prepare_spacy_data

# TODO: implelent rule based approach
class Rulse_based_model:
    def __init__(self, checkpoint_path):
        self._checkpoint_path = checkpoint_path

    
    def _postprocess(self, doc: Doc) -> list:
        tag_dict = {"GPE" : "LOC",
                    "LOC" : "LOC",
                    "PERSON" : "PER",
                    "ORG" : "ORG",
                    "LANGUAGE" : "MISC",
                    "WORK_OF_ART" : "MISC"}
        res_list = []
        for token in doc:
            token_name = token.ent_type_
            if token_name in tag_dict.keys():
                res_list.append( token.ent_iob_ + "-" + tag_dict[token_name])
            else:
                res_list.append("O")
        return res_list


    def predict(self, X) -> list:
        if not os.path.isfile(self._checkpoint_path):
            os.mknod(self._checkpoint_path)
        
        nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        res_list = []
        for sentence in X['Sentence']:
            doc = nlp(Doc(nlp.vocab, words=sentence))
            res_list.append(self._postprocess(doc))

        return res_list

