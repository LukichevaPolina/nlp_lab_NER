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