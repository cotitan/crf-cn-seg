import os
import json

def build_vocab(filein, vocab_file):
    print("Building vocabulary...")
    fin = open(filein)
    vocab = {"<s>": 0, "</s>": 1}
    tag2id = {"<s>": 0, "</s>": 1}
    for _, line in enumerate(fin):
        if line.strip() == "":
            continue
        ch, tag = line.strip().split()
        if ch not in vocab:
            vocab[ch] = len(vocab)
        if tag not in tag2id:
            tag2id[tag] = len(tag2id)
    fin.close()
    json.dump([vocab, tag2id], open(vocab_file, "w"))

def load_data(filein, vocab, tag2id):
    print("Loading data...")
    fin = open(filein)
    X = []
    Y = []
    x = []
    y = []
    for _, line in enumerate(fin):
        if line.strip() == "":
            X.append(x)
            Y.append(y)
            x = []
            y = []
        else:
            try:
                char, tag = line.strip().split()
                x.append(vocab[char])
                y.append(tag2id[tag])
            except:
                print(line)
    return X, Y

class BatchManager:
    def __init__(self, datas, batch_size):
        self.steps = int(len(datas) / batch_size)
        self.datas = datas
        self.bs = batch_size
        self.bid = 0

    def next_batch(self):
        batch = list(self.datas[self.bid * self.bs: (self.bid + 1) * self.bs])
        self.bid += 1
        if self.bid == self.steps:
            self.bid = 0
        return batch


if __name__ == "__main__":
    X, Y = load_data("train.bioes", "vocab.json")
    pass
    