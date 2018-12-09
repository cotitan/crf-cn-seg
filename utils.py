import os
import json

vocab_file = "vocab.json"
tag2id = {"B": 0, "I": 1, "O": 2, "E": 3, "S": 4, "<s>": 5, "</s>": 6}

def build_vocab(filein, vocab_file):
    fin = open(filein)
    vocab = {"<s>": 0, "</s>": 1}
    for _, line in enumerate(fin):
        for word in line.strip().split():
            for ch in word:
                if ch not in vocab:
                    vocab[ch] = len(vocab)
    fin.close()
    json.dump(vocab, open(vocab_file, "w"))

def load_data(filein, vocab_file):
    if not os.path.exists(vocab_file):
        build_vocab(filein, vocab_file)
    vocab = json.load(open(vocab_file))

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
    return X, Y, vocab, tag2id

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
    X, Y = load_data("train.bioes")
    pass
    