import numpy as np
import torch
from torch import nn
import utils

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
    def __init__(self, num_tags, vocab):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.vocab = vocab

        self.emit_score = nn.Parameter(torch.randn(len(vocab), num_tags).cuda())
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags).cuda())
        # never transfer to start tag or never transfer from stop tag
        # self.transitions[:, tag2id["<s>"]] = -10000.
        # self.transitions[tag2id["</s>"], :] = -10000.
    
    def forward_alg(self, x):
        init_alphas = torch.ones(self.num_tags).cuda() * -10000
        init_alphas[tag2id["<s>"]] = 0

        alphas = init_alphas

        for ch_id in x[1:]:
            emit_score = self.emit_score[ch_id].view(1, -1)
            _score = alphas.view(-1, 1) + emit_score + self.transitions
            # _score = alphas + emit_score
            for i in range(self.num_tags):
                alphas[i] = log_sum_exp(_score[:, i]).squeeze()
        alphas = alphas + self.transitions[:, self.vocab["</s>"]]
        z = log_sum_exp(alphas)
        return z

    def neg_log_likelihood(self, x, y):
        return self.forward_alg(x) - self.score(x, y)
    
    def score(self, x, y):
        s = 0
        for i in range(1, len(x)-1):
            s += self.emit_score[x[i]][y[i]]
        for i in range(len(y)-1):
            s += self.transitions[y[i]][y[i+1]]
        return s

    def infer(self, x):
        pass
    

if __name__ == "__main__":
    X, Y, vocab, tag2id = utils.load_data("train.bioes", "vocab.json")
    model = CRF(len(tag2id), vocab).cuda()
    # model.load_state_dict(torch.load(model_file))
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

