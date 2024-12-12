import os
from pathlib import Path

from tqdm import tqdm
import spacy
from spacy.tokens import Doc
from spacy.pipeline import EntityRuler

from src.models.NER_lists import cities, countries, female_names, male_names, surnames, organizations


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

    def _match_tags(self, doc_pred_list, tags_true_list):
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

    def _spacy_pred(self, X):
        res_list = []
        for idx, sentence in enumerate(X):
            doc = self._nlp(Doc(self._nlp.vocab, words=sentence))
            res_list.append(doc)

        return res_list

    def _add_custom_rules(self, X, y):
        ruler = self._nlp.add_pipe("entity_ruler")

        patterns = [{"label": "LOC", "pattern": [{"LOWER": loc}]}
                    for loc in cities]
        patterns += [{"label": "LOC", "pattern": [{"LOWER": loc}]}
                     for loc in countries]
        patterns += [{"label": "PER", "pattern": [{"LOWER": per}]}
                     for per in female_names]
        patterns += [{"label": "PER", "pattern": [{"LOWER": per}]}
                     for per in male_names]
        patterns += [{"label": "PER", "pattern": [{"LOWER": per}]}
                     for per in surnames]
        patterns += [{"label": "ORG", "pattern": [{"LOWER": per}]}
                     for per in organizations]
        ruler.add_patterns(patterns)

    def fit(self, X, y):
        if self._is_use_custom_rules:
            self._add_custom_rules(X, y)
        res_list = self._spacy_pred(X)
        self.tags_dict = self._match_tags(res_list, y)

    def predict(self, X) -> list:
        res_list = self._spacy_pred(X)
        res_list = self._postprocess(res_list)

        return res_list
