import torch
from torch import nn

torch.manual_seed(1)

def log_sum_exp2d(mat):
    maxx = torch.max(mat, dim=0)[0].view(1, -1)
    return maxx + torch.log(torch.sum(torch.exp(mat-maxx), dim=0))

def log_sum_exp(mat):
    maxx = torch.max(mat).view(1, -1)
    return maxx + torch.log(torch.sum(torch.exp(mat - maxx)))

class CRF(nn.Module):
    def __init__(self, vocab, tag2id, device):
        super(CRF, self).__init__()
        self.num_tags = len(tag2id)
        self.vocab = vocab
        self.tag2id = tag2id
        self.device = device

        # make sure emit_score and transitions is leaf
        # that is, the is_leaf attribute of them is True
        # or the model will not converage
        self.emit_score = nn.Parameter(torch.randn(len(vocab), self.num_tags))

        # never transfer to START_TAG or never transfer from STOP_TAG
        transitions = torch.randn(self.num_tags, self.num_tags)
        transitions[:, tag2id["<s>"]] = -10000.
        transitions[tag2id["</s>"], :] = -10000.
        self.transitions = nn.Parameter(transitions)
    
    def forward_alg(self, x):
        init_alphas = torch.ones(self.num_tags).to(self.device) * -10000
        init_alphas[self.tag2id["<s>"]] = 0

        # init_alphas = torch.zeros(self.num_tags).to(self.device)
        # init_alphas[self.tag2id["<s>"]] = 1

        alphas = init_alphas

        for ch_id in x:
            emit_score = self.emit_score[ch_id].view(1, -1)
            _score = alphas.view(-1, 1) + self.transitions + emit_score
            alphas = log_sum_exp2d(_score).squeeze()
            # for i in range(self.num_tags):
            #     alphas[i] = log_sum_exp(_score[:, i]).squeeze()
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
        res = res[:-1]
        return reversed(res)
    
    def infer(self, x):
        init_alphas = torch.ones(self.num_tags).to(self.device) * -10000
        init_alphas[self.tag2id["<s>"]] = 0

        path = []
        alphas = init_alphas

        for ch_id in x:
            emit_score = self.emit_score[ch_id].view(1, -1)
            _score = alphas.view(-1, 1) + emit_score + self.transitions # 7*7
            alphas, maxx_id = torch.max(_score, dim=0)
            path.append(maxx_id)
        alphas += self.transitions[:, self.tag2id["</s>"]]

        return self.back_trace(path, torch.argmax(alphas))
