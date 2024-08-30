import json
import numpy as np
import pandas as pd
from collections import defaultdict

# def build_vocab(filein, vocab_file):
#     print("Building vocabulary...")
#     fin = open(filein)
#     vocab = {"<pad>": 0, "<unk>": 1}
#     # tag2id = {"<pad>": 0}
#     # vocab = {"<unk>": 0}
#     tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
#     for _, line in enumerate(fin):
#         if line.strip() == "":
#             continue
#         ch, tag = line.strip().split()
#         if ch not in vocab:
#             vocab[ch] = len(vocab)
#         if tag not in tag2id:
#             tag2id[tag] = len(tag2id)
#     fin.close()
#     json.dump([vocab, tag2id], open(vocab_file, "w"))


def build_vocab(filein, vocab_file, tag_schema='bmes'):
    print("Building vocabulary...")
    fin = open(filein)
    vocab = {"<pad>": 0, "<unk>": 1}
    if tag_schema == 'bmes':
        tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    elif tag_schema == 'bio':
        tag2id = {'B': 0, 'I': 1, 'O': 2}
    else:
        raise Exception("Unknown tag schema")

    word_cnt = defaultdict(int)

    for _, line in enumerate(fin):
        if line.strip() == "":
            continue
        ch, tag = line.strip().split()
        word_cnt[ch] += 1
        if tag not in tag2id:
            tag2id[tag] = len(tag2id)
            
    df_cnt = pd.DataFrame([[k,v] for k,v in word_cnt.items()], columns=['word', 'cnt'])
    df_cnt = df_cnt.sort_values(by='cnt', ascending=False).reset_index(drop=True)
    for _,row in df_cnt.query('cnt>0').iterrows():
        vocab[row['word']] = len(vocab)

    fin.close()
    json.dump([vocab, tag2id], open(vocab_file, "w"))


def load_data(filein, vocab, tag2id, max_len=32):
    print("Loading data...")
    fin = open(filein)
    X = []
    Y = []
    x = []
    y = []
    masks = []
    for _, line in enumerate(fin):
        if line.strip() == "":
            if len(x) == 0:
                continue
            if len(x) > max_len:
                nblock = int(np.ceil(len(x)/max_len))
                k, m = divmod(len(x), nblock)

                for i in range(nblock):
                    st, ed = i * k + min(i, m), (i+1)*k + min(i+1, m)

                    len_fill = max_len - (ed - st)
                    X.append(x[st:ed] + [vocab['<pad>']] * len_fill)
                    Y.append(y[st:ed] + [0] * len_fill)

                    masks.append([1] * (ed-st) + [0] * len_fill)
            else:
                len_fill = max_len - len(x)
                X.append(x + [vocab['<pad>']] * len_fill)
                Y.append(y + [0] * len_fill)

                masks.append([1] * len(x) + [0] * len_fill)
            x = []
            y = []
        else:
            try:
                char, tag = line.strip().split()
                x.append(vocab[char if char in vocab else "<unk>"])
                y.append(tag2id[tag])
            except:
                print(line)
    return X, Y, masks


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
    