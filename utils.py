import json
import numpy as np

def build_vocab(filein, vocab_file, n_examples=20000):
    print("Building vocabulary...")
    fin = open(filein)
    # vocab = {"<pad>": 0, "<unk>": 1}
    # tag2id = {"<pad>": 0}
    vocab = {"<unk>": 0}
    tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    eg_cnt = 0
    for _, line in enumerate(fin):
        if line.strip() == "":
            eg_cnt += 1
            if eg_cnt >= n_examples:
                break
            continue
        ch, tag = line.strip().split()
        if ch not in vocab:
            vocab[ch] = len(vocab)
        if tag not in tag2id:
            tag2id[tag] = len(tag2id)
    fin.close()
    json.dump([vocab, tag2id], open(vocab_file, "w"))


def load_data(filein, vocab, tag2id, n_examples=20000):
    print("Loading data...")
    fin = open(filein)
    X = []
    Y = []
    x = []
    y = []
    eg_cnt = 0
    for i, line in enumerate(fin):
        if line.strip() == "":
            X.append(x)
            Y.append(y)
            x = []
            y = []
            eg_cnt += 1
            if eg_cnt >= n_examples:
                break
        else:
            try:
                char, tag = line.strip().split()
                x.append(vocab[char if char in vocab else "<unk>"])
                y.append(tag2id[tag])
            except:
                print(line)
    return X, Y


class BatchManager:
    def __init__(self, X, Y, batch_size):
        self.steps = int(len(X) / batch_size)
        self.X = X
        self.Y = Y
        self.bs = batch_size
        self.bid = 0
        perm = np.random.permutation(len(self.X))
        self.X, self.Y = [self.X[p] for p in perm], [self.Y[p] for p in perm]

    def next_batch(self):
        st, ed = self.bid * self.bs, (self.bid + 1) * self.bs
        x = self.X[st:ed]
        y = self.Y[st:ed]
        
        max_len = max(len(line) for line in x)
        mask = [[1] * len(line) + [0] * (max_len - len(line)) for line in x]
        x = [line + [0] * (max_len - len(line)) for line in x]
        y = [line + [0] * (max_len - len(line)) for line in y]

        self.bid += 1
        if self.bid == self.steps:
            self.bid = 0
            perm = np.random.permutation(len(self.X))
            self.X, self.Y = [self.X[p] for p in perm], [self.Y[p] for p in perm]
        return x, y, mask


if __name__ == "__main__":
    vocab, tag2id = json.load(open("data/vocab.json"))
    X, Y = load_data("data/train.bmes", vocab, tag2id)
    print("")
    