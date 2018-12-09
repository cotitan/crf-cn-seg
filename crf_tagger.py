import numpy as np
import torch
from torch import nn
import os
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default="models/params_0.pkl")
args = parser.parse_args()

torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp1(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp(mat):
    maxx = torch.max(mat).view(1, -1)
    return maxx + torch.log(torch.sum(torch.exp(mat - maxx)))

class CRF(nn.Module):
    def __init__(self, vocab, tag2id):
        super(CRF, self).__init__()
        self.num_tags = len(tag2id)
        self.vocab = vocab
        self.tag2id = tag2id

        self.emit_score = nn.Parameter(torch.randn(len(vocab), self.num_tags).cuda())

        transitions = torch.randn(self.num_tags, self.num_tags)
        transitions[:, tag2id["<s>"]] = -10000.
        transitions[tag2id["</s>"], :] = -10000.
        
        self.transitions = nn.Parameter(transitions).cuda()
        # never transfer to START_TAG or never transfer from STOP_TAG
    
    def forward_alg(self, x):
        init_alphas = torch.ones(self.num_tags).cuda() * -10000
        init_alphas[self.tag2id["<s>"]] = 0

        alphas = init_alphas

        for ch_id in x:
            emit_score = self.emit_score[ch_id].view(1, -1)
            _score = alphas.view(-1, 1) + emit_score + self.transitions
            # _score = alphas + emit_score
            for i in range(self.num_tags):
                alphas[i] = log_sum_exp(_score[:, i]).squeeze()
        alphas = alphas + self.transitions[:, self.tag2id["</s>"]]
        z = log_sum_exp(alphas)
        return z

    def neg_log_likelihood(self, x, y):
        return self.forward_alg(x) - self.score(x, y)
    
    def score(self, x, y):
        s = 0
        for i in range(len(x)):
            s += self.emit_score[x[i]][y[i]]

        s += self.transitions[self.tag2id["<s>"]][y[0]]
        for i in range(len(y) - 1):
            s += self.transitions[y[i]][y[i+1]]
        s += self.transitions[y[-1]][self.tag2id["</s>"]]

        return s

    def back_trace(self, path, start_id):
        res = [start_id]
        for tags in reversed(path):
            res.append(tags[res[-1]])
        return reversed(res)
    
    def infer(self, x):
        init_alphas = torch.ones(self.num_tags).cuda() * -10000
        init_alphas[self.tag2id["<s>"]] = 0

        path = []
        alphas = init_alphas

        for ch_id in x:
            emit_score = self.emit_score[ch_id].view(1, -1)
            _score = alphas.view(-1, 1) + emit_score + self.transitions # 7*7
            alphas, maxx_id = torch.max(_score, dim=0)[0]
            path.append(maxx_id)
        alphas += self.transitions[:, self.tag2id["</s>"]]

        return self.back_trace(path, torch.argmax(alphas))


if __name__ == "__main__":
    X, Y, vocab, tag2id = utils.load_data("train.bioes", "vocab.json")
    model = CRF(vocab, tag2id).cuda()
    if os.path.exists(args.model_file):
        model.load_state_dict(torch.load(args.model_file))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    batch_size = 32
    batchX = utils.BatchManager(X, batch_size)
    batchY = utils.BatchManager(Y, batch_size)

    n_epochs = 10
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

            print(bid, loss)
        model.cpu()
        torch.save(model.state_dict(), 'models/params_%d.pkl' % epoch)
        model.cuda()
        # if i % 10 == 0:
        #     print(i, loss.detach().numpy())

