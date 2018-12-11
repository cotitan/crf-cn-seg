import numpy as np
import torch
from torch import nn
import os
import utils
import argparse
import json
from crf import CRF

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default="models/params_0.pkl")
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--train_file', type=str, default="data/train.bmes")
parser.add_argument('--test_file', type=str, default="")
parser.add_argument('--vocab_file', type=str, default="data/vocab.json")
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

vocab_file = args.vocab_file
train_file = args.train_file
model_file = args.model_file
_test_file = args.test_file
batch_size = args.batch_size
n_epochs = args.n_epochs


def test_infer(model, vocab, tag2id):
    id2tag = {v:k for k,v in tag2id.items()}
    with torch.no_grad():
        sentence = "中山大学创办于1924年，是孙中山先生一手创立的"
        print(sentence)
        x = [vocab[ch] for ch in sentence]
        ids = model.infer(x)
        ids = [int(x.cpu().numpy()) for x in ids]
        tags = [id2tag[i] for i in ids]
        print(tags)
        for i in range(len(sentence)):
            if tags[i] == "E" or tags[i] == "S":
                print("%s " % sentence[i], end="")
            else:
                print(sentence[i], end="")
        print("")

def evaluate_on_file(model, vocab, tag2id, filein, fileout):
    fin = open(filein)
    fout = open(fileout, "w")
    id2tag = {v:k for k,v in tag2id.items()}
    with torch.no_grad():
        for sentence in fin:
            sentence = sentence.strip()
            x = [vocab.get(ch, vocab["<unk>"]) for ch in sentence]
            ids = model.infer(x)
            ids = [int(x.cpu().numpy()) for x in ids]
            tags = [id2tag[i] for i in ids]
            for i in range(len(sentence)):
                if tags[i] == "E" or tags[i] == "S":
                    fout.write("%s  " % sentence[i])
                else:
                    fout.write(sentence[i])
            fout.write("\n")
    fin.close()
    fout.close()

def train(model, train_file, vocab, tag2id):
    # load train data
    X, Y = utils.load_data(train_file, vocab, tag2id)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)

    batchX = utils.BatchManager(X, batch_size)
    batchY = utils.BatchManager(Y, batch_size)

    for epoch in range(n_epochs):
        for bid in range(batchX.steps):
            optimizer.zero_grad()
            x = batchX.next_batch()
            y = batchY.next_batch()
            loss = torch.zeros(1).cuda()
            for i in range(len(x)):
                x[i] = torch.tensor(x[i]).cuda()
                y[i] = torch.tensor(y[i]).cuda()
                loss += model.neg_log_likelihood(x[i], y[i]).squeeze()
            # loss.backward(retain_graph=True)
            loss /= batch_size
            loss.backward()
            optimizer.step()

            print("epoch %d, step %f" % (
                bid, loss.detach().cpu().numpy().squeeze() ))
        scheduler.step()
        model.cpu()
        torch.save(model.state_dict(), 'models/params_%d.pkl' % epoch)
        model.cuda()

if __name__ == "__main__":
    # load vocab, tag2id
    if not os.path.exists(args.vocab_file):
        utils.build_vocab(train_file, vocab_file)
    vocab, tag2id = json.load(open(vocab_file))

    print(tag2id)
    # load model
    model = CRF(vocab, tag2id).cuda()
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
    
    # test
    if args.test:
        test_infer(model, vocab, tag2id)
        evaluate_on_file(model, vocab,tag2id, "data/pku_test.utf8", "data/pku_test.out")
        exit(0)

    # train
    train(model, train_file, vocab, tag2id)
