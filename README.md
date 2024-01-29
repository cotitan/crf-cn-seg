# crf-cn-seg
a simple Chinese word segmentation tool based on CRF, implemented in pytorch

## directories
```
.
├── build_dataset.py  # Transfer raw text into labeled data
├── crf.py  # pytorch implementation of CRF
├── data
│   ├── pku_training.utf8  # raw training file
│   ├── train.bmes  # labeled training file
│   ├── vocab.json
│   ├── pku_test.utf8  # raw test file
│   ├── pku_test_gold.utf8
│   ├── pku_training_words.utf8  # used for evaluate
│   └── pku_test.out  # my predictions
├── main.py
├── models.py # class CRF, BiLSTMCRF, TransformerCRF
├── models
│   └── params_0.pkl
├── README.md
├── scripts
│   ├── mwseg.pl
│   └── score  # used to evaluate model's performance
└── utils.py
```

## train
```
python main.py  
```
Generally, you will get 0.77+ F1 after 2 epochs.

## evaluate
* generate segmented file
```
python main --test --model_file models/params_1.pkl
```
* evaluate p, r, f using perl scripts
```
./scripts/score data/pku_training_words.utf8 data/pku_test_gold.utf8 data/pku_test.out > score.utf8
```
At the end of score.utf8, you will find the precision, recall, F1.

## performance

|models|precision|recall|F1-measure|
|--|--|--|--|
|CRF (3 epochs)|0.796|0.804|0.800|
|BiLSTM+CRF (1 epoch)|0.876|0.888|0.882|
|Transformer+CRF (3 epochs)|0.785|0.797|0.791|