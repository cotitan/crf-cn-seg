# crf-cn-seg
a simple Chinese word segmentation tool based on CRF, implemented in pytorch

## directories
```
.
├── build_dataset.py  # Transfer raw text into labeled data
├── crf.py  # model file
├── data
│   ├── pku_training.utf8  # raw training file
│   ├── train.bmes  # labeled training file
│   ├── vocab.json
│   ├── pku_test.utf8  # raw test file
│   ├── pku_test_gold.utf8
│   ├── pku_training_words.utf8  # used for evaluate
│   └── pku_test.out  # my predictions
├── main.py
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
python main --test True --model_file models/params_1.pkl
```
* evaluate p, r, f using perl scripts
```
./scripts/score data/pku_training_words.utf8 data/pku_test_gold.utf8 data/pku_test.out > score.utf8
```
At the end of score.utf8, you will find the precision, recall, F1.

## performance

|epoch|precision|recall|F1-measure|
|--|--|--|--|
|0|0.732|0.759|0.745|
|1|0.766|0.786|0.776|
|2|0.769|0.787|0.778|
|3|0.776|0.786|0.781|