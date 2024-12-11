import os
from pathlib import Path

from tqdm import tqdm
import spacy
from spacy.tokens import Doc
from spacy.pipeline import EntityRuler


from src.preprocessing.preprocessing import prepare_spacy_data


class Rulse_based_model:
    def __init__(self, is_use_custom_rules=False):
        self._nlp = spacy.load("en_core_web_sm", disable=[
                               "tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        self._is_use_custom_rules = is_use_custom_rules
        self.tags_dict = None

    def _postprocess(self, doc_list: Doc) -> list:
        # convert spacy predicted tags to target tags
        res_list = []
        for doc in doc_list:
            tags_list = []
            for token in doc:
                token_name = token.ent_iob_ + "-" + token.ent_type_
                tags_list.append(self.tags_dict[token_name])
            res_list.append(tags_list)
        return res_list

    def _match_tags(self, doc_pred_list: list, tags_true_list: list) -> dict:
        # get dict with the most popular target tag for spacy tags
        target_values = {"O": 0, "B-MISC": 0, "I-MISC": 0, "B-PER": 0,
                         "I-PER": 0, "B-LOC": 0, "I-LOC": 0, "B-ORG": 0, "I-ORG": 0, }
        tags_dict = {}
        for doc_pred, tags_true in zip(doc_pred_list, tags_true_list):
            for idx, el in enumerate(doc_pred):
                tag_pred = el.ent_iob_ + "-" + el.ent_type_
                tag_true = tags_true[idx]
                if tag_pred in tags_dict.keys():
                    tags_dict[tag_pred][tag_true] += 1
                else:
                    tags_dict[tag_pred] = target_values.copy()

        res_dict = {}
        for key, val in tags_dict.items():
            res_dict[key] = max(val, key=val.get)

        return res_dict

    def _spacy_pred(self, X) -> list:
        res_list = []
        for idx, sentence in enumerate(X):
            doc = self._nlp(Doc(self._nlp.vocab, words=sentence))
            res_list.append(doc)

        return res_list

    def _add_custom_rules(self, X, y) -> None:
        ruler = self._nlp.add_pipe("entity_ruler")
        locations = set()
        organizations = set()
        misc_entities = set()
        persons = set()
        for sentance, tags in zip(X, y):
            for idx, tag in enumerate(tags):
                if tag == "B-LOC" or tag == "I-LOC":
                    locations.add(sentance[idx])
                elif tag == "B-ORG" or tag == "I-ORG":
                    organizations.add(sentance[idx])
                elif tag == "B-MISC" or tag == "I-MISC":
                    misc_entities.add(sentance[idx])
                elif tag == "B-PER" or tag == "I-PER":
                    persons.add(sentance[idx])

        patterns = [{"label": "LOC", "pattern": [{"LOWER": loc}]}
                    for loc in locations]
        patterns += [{"label": "ORG", "pattern": [{"LOWER": org}]}
                     for org in organizations]
        patterns += [{"label": "MISC", "pattern": [{"LOWER": misc}]}
                     for misc in misc_entities]
        patterns += [{"label": "PER", "pattern": [{"LOWER": per}]}
                     for per in persons]
        ruler.add_patterns(patterns)

    def fit(self, X, y) -> None:
        if self._is_use_custom_rules:
            self._add_custom_rules(X, y)
        res_list = self._spacy_pred(X)
        self.tags_dict = self._match_tags(res_list, y)

    def predict(self, X) -> list:
        res_list = self._spacy_pred(X)
        res_list = self._postprocess(res_list)

        return res_list
