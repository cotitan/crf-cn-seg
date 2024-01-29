import numpy as np
import torch
from torch import nn
import os
import utils
import argparse
import json
from models import VanillaCRF, BiLSTMCRF, TransformerCRF

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default="ckpts/params_0.pkl")
parser.add_argument('--test', action='store_true')
parser.add_argument('--train_file', type=str, default="data/train.bmes")
parser.add_argument('--test_file', type=str, default="")
parser.add_argument('--vocab_file', type=str, default="data/vocab.json")
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

vocab_file = args.vocab_file
train_file = args.train_file
model_file = args.model_file
_test_file = args.test_file
batch_size = args.batch_size
n_epochs = args.n_epochs


def test_infer(model, vocab, tag2id, device=torch.device('cpu')):
    id2tag = {v:k for k,v in tag2id.items()}
    with torch.no_grad():
        sentence = "中山大学创办于1924年，是孙中山先生一手创立的"
        # sentence = "南京市长江大桥"
        print(sentence)
        x = torch.LongTensor([[vocab[ch] for ch in sentence]]).to(device)
        mask = torch.LongTensor([[1] * x.shape[1]]).to(device)
        ids = model.infer(x, mask=mask)[0]
        # ids = [int(x.cpu().numpy()) for x in ids]
        tags = [id2tag[i] for i in ids]
        for i in range(len(sentence)):
            if tags[i] == "E" or tags[i] == "S":
                print("%s " % sentence[i], end="")
            else:
                print(sentence[i], end="")
        print()

def evaluate_on_file(model, vocab, tag2id, filein, fileout, device=torch.device('cpu')):
    fin = open(filein)
    fout = open(fileout, "w")
    id2tag = {v:k for k,v in tag2id.items()}
    with torch.no_grad():
        for sentence in fin:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            x = torch.LongTensor([[vocab.get(ch, vocab["<unk>"]) for ch in sentence]]).to(device)
            mask = torch.BoolTensor([[1] * x.shape[1]]).to(device)
            try:
                ids = model.infer(x, mask)[0]
                # ids = [int(x.cpu().numpy()) for x in ids]
                tags = [id2tag[i] for i in ids]

                for i in range(len(sentence)):
                    if tags[i] == "E" or tags[i] == "S":
                        fout.write("%s  " % sentence[i])
                    else:
                        fout.write(sentence[i])
                fout.write("\n")
            except:
                print(len(sentence), sentence)
                print(len(x), x)
                print(len(ids), ids)
    fin.close()
    fout.close()


def train(model, train_file, vocab, tag2id, device, ckpts):
    # load train data
    X, Y = utils.load_data(train_file, vocab, tag2id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)

    loader = utils.BatchManager(X, Y, batch_size)

    for epoch in range(n_epochs):
        for bid in range(loader.steps):
            optimizer.zero_grad()
            x, y, mask = loader.next_batch()
            # loss = torch.zeros(1).to(device)
            # for i in range(len(x)):
            #     x[i] = torch.tensor(x[i]).to(device)
            #     y[i] = torch.tensor(y[i]).to(device)
            #     loss += model.neg_log_likelihood(x[i], y[i]).squeeze()
            # loss.backward(retain_graph=True)

            x, y, mask = [torch.LongTensor(k).to(device) for k in [x,y,mask]]
            loss = model(x, y, mask.bool())
            # loss = loss.mean()

            loss.backward()
            optimizer.step()
            ckpts['trn_loss'].append(loss.item())

            print("epoch %d/%d, step %d/%d, loss=%f" % (epoch, n_epochs, bid, loader.steps,
                   loss.detach().cpu().numpy().squeeze()))
        scheduler.step()
        torch.save(model.state_dict(), 'ckpts/params_%d.pkl' % epoch)
        

if __name__ == "__main__":
    # load vocab, tag2id
    if not os.path.exists(args.vocab_file):
        utils.build_vocab(train_file, vocab_file)
    vocab, tag2id = json.load(open(vocab_file))

    print(tag2id)
    # load model
    device = torch.device('cpu')
    # model = VanillaCRF(vocab, tag2id, device).to(device)
    model = BiLSTMCRF(vocab, tag2id, 32, 64, device).to(device)
    # model = TransformerCRF(vocab, tag2id, 16, 16, device).to(device)
    for k, p in model.named_parameters():
        print(k, p.shape)

    if os.path.exists(model_file) and args.test:
        model.load_state_dict(torch.load(model_file))
    
    # test
    if args.test:
        test_infer(model, vocab, tag2id)
        evaluate_on_file(model, vocab,tag2id, "data/pku_test.utf8", "data/pku_test.out")
        exit(0)

    # train
    train(model, train_file, vocab, tag2id, device)
