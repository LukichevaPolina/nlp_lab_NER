# Named entity recognition lab

## Prerequests
> Python3.12 is used

1. Clone the reposiroty
```bash
git clone https://github.com/LukichevaPolina/nlp_lab_2.git
cd nlp_lab
```

2. Install requirements.txt
```bash
pip3 install -r requirements.txt
```

3. Set up `PYTHONPATH`
```bash
export PYTHONPATH=$PYTHONPATH:$PWD
```

## EDA
Dataset is taken from [github](https://github.com/Babelscape/wikineural/tree/master/data/wikineural/en). The Babelscape/wikineural NER Dataset is a comprehensive and diverse collection of multilingual text data specifically designed for the task of Named Entity Recognition (NER). It offers an extensive range of labeled sentences in nine different languages: French, German, Portuguese, Spanish, Polish, Dutch, Russian, English, and Italian. *We used only English language files*. Dataset consist of three columns with `position in sentence`, `tokens` and `tags`.  
* The tokens column contains the individual words or characters in each labeled sentence. 
* The ner_tags column provides named entity recognition tags for each token, indicating their entity types.
The `tags` is our target, which could take nine different values.  

Dataset contains 3 files:
* The `train.conllu` has 92719 sentences.
* The `test.conllu` has 11596 sentences.
* The `val.conllu` has 11589 sentences.


| Train | Test | Val |
:---------------:|:--------------:|:---------:
![alt text](./plots/sentence_length_distribution_train.png) | ![alt text](./plots/sentence_length_distribution_test.png) | ![alt text](./plots/sentence_length_distribution_val.png)


| Train | Test | Val |
:---------------:|:--------------:|:---------:
![alt text](./plots/tags_distribution_train_O_tag_False.png) | ![alt text](./plots/tags_distribution_test_O_tag_False.png) | ![alt text](./plots/tags_distribution_val_O_tag_False.png)


| Train | Test | Val |
:---------------:|:--------------:|:---------:
![alt text](./plots/tag_word_position_distribution_train.png) | ![alt text](./plots/tag_word_position_distribution_test.png) | ![alt text](./plots/tag_word_position_distribution_val.png)


## Preprocessing
Consider removing any unnecessary punctuation marks or special characters unless they carry significant meaning in certain languages.


## Rule based models
### Using Spacy model
|             | precision |   recall | f1-score |  support |
|--------     | --------  | -------- | -------- |  ------- | 
|         LOC |     0.56  |    0.70  |    0.62  |   15925  |
|        MISC |     0.16  |    0.73  |    0.27  |   83014  |
|         ORG |     0.41  |    0.31  |    0.35  |   27299  |
|         PER |     0.68  |    0.67  |    0.67  |    5226  |
|             |           |          |          |          |  
|   micro avg |     0.46  |    0.58  |    0.51  |  131464  |
|   macro avg |     0.45  |    0.60  |    0.48  |  131464  |
|weighted avg |     0.52  |    0.58  |    0.53  |  131464  |

**f1-score:** 0.5130712489048413

### Add custom rules
* Adding all tags from train dataset
|             | precision |   recall | f1-score |  support |
|--------     | --------  | -------- | -------- |  ------- | 
|         LOC |     0.56  |    0.21  |    0.30  |   15925  |
|        MISC |     0.17  |    0.01  |    0.02  |   83014  |
|         ORG |     0.41  |    0.05  |    0.09  |   27299  |
|         PER |     0.68  |    0.67  |    0.67  |    5226  |
|             |           |          |          |          |  
|   micro avg |     0.46  |    0.07  |    0.12  |  131464  |
|   macro avg |     0.45  |    0.24  |    0.27  |  131464  |
|weighted avg |     0.29  |    0.07  |    0.10  |  131464  |

**f1-score:** 0.12061824293030213

* Add popular words from train dataset
