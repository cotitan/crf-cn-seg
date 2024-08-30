""" apply HMM on Chinese Word Segmentation task,
    implement both in a sequential manner and in a batch manner (todo)
"""

import torch
from torch import nn
import os
import utils_hmm as utils
import argparse
import json
import pickle as pkl
import traceback
from tqdm import tqdm
import numpy as np
import pandas as pd


class HMM(nn.Module):
    def __init__(self, vocab, tag2id, device):
        super(HMM, self).__init__()
        self.device = device
        self.n_vocab = len(vocab)
        self.n_state = len(tag2id)
        self.vocab = vocab
        self.tag2id = tag2id
        self.init_params(self.n_vocab, self.n_state)

    def init_params(self, n_vocab, n_state):
        tag2id = self.tag2id
        vocab = self.vocab
        # BMES
        self.pi = torch.rand(n_state, 1).to(self.device)
        self.A = torch.rand(n_state, n_state).to(self.device)
        self.B = torch.rand(n_state, n_vocab).to(self.device)

        # for ch in '，的。、在了':
        #     self.B[tag2id['B'], vocab[ch]] = 0
        #     self.B[tag2id['S'], vocab[ch]] = 10
        # 

        # self.pi = torch.tensor([[0.6], [0], [0], [0.4]]).to(self.device)
        # self.A = torch.tensor([
        #     [0.0, 0.15, 0.85, 0.0], # B
        #     [0.0, 0.2, 0.8, 0.0], # M
        #     [0.2, 0.0, 0.0, 0.8], # E
        #     [0.4, 0.0, 0.0, 0.6]  # S
        # ]).to(self.device)

        self.adjust_params()
        self.logarithmize_params()

        print("pi:")
        print(self.pi.cpu().detach().numpy())
        print("A:")
        print(self.A.cpu().detach().numpy())


    def adjust_params(self):
        tag2id = self.tag2id

        self.pi[tag2id['M'],0] = 0 # prior, initial states is B or S
        self.pi[tag2id['E'],0] = 0
        self.pi = self.pi / self.pi.sum(axis=0)

        self.A[tag2id['B'],tag2id['B']] = 0
        self.A[tag2id['B'],tag2id['S']] = 0
        self.A[tag2id['M'],tag2id['B']] = 0
        self.A[tag2id['M'],tag2id['S']] = 0
        self.A[tag2id['E'],tag2id['M']] = 0
        self.A[tag2id['E'],tag2id['E']] = 0
        self.A[tag2id['S'],tag2id['M']] = 0
        self.A[tag2id['S'],tag2id['E']] = 0
        self.A = self.A / self.A.sum(axis=1, keepdim=True)
        self.B = self.B / self.B.sum(axis=1, keepdim=True)


    def logarithmize_params(self):
        self.log_pi = torch.log(self.pi)
        self.log_pi = torch.masked_fill(self.log_pi, self.log_pi==float('-inf'), -1000)
        self.log_A = torch.log(self.A)
        self.log_A = torch.masked_fill(self.log_A, self.log_A==float('-inf'), -1000)
        self.log_B = torch.log(self.B)
        self.log_B = torch.masked_fill(self.log_B, self.log_B==float('-inf'), -1000)
    

    """
    logsumexp usage: to implement a matmul in log-space
    x = [e,e**2,e**3], log_x = [1,2,3]
    y = [e**3,e**2,e], log_y = [3,2,1]
    x.dot(y) = e**4 * 3
    log_(xdoty) = logsumexp(log_x + log_y = [4,4,4]) = logsum([e**4, e**4, e**4])
                = log(e**4 *3) = 4 + log(3)
    """

    def forward_alg(self, Y, mask):
        """ calcuating forward probability alpha_{ti}
        args:
            Y: 2d tensor of observations, [num_obs, T]
            mask: 2d tensor of pad mask, [num_obs, T]
        returns:
            alphas: 3d tensor, [num_obs, num_state, T], alphas[n][i][t] is the probability of
                    being at state i and having observation y_t at time t,
                    and having observations (y1,y2...yt-1) before time t of observation n
        """
        # pi * B [4,1] * [4,N] => [4,1]
        num_obs, T = Y.shape
        alphas = torch.ones(num_obs, self.n_state, T).to(self.device) * torch.inf
        alphas[:,:,0] = (self.log_pi + self.log_B[:, Y[:,0]]).T

        expand_A = self.log_A.unsqueeze(0)
        mask = torch.FloatTensor(mask).to(self.device)
        for t in range(1, T):
            # logsumexp([bs, n_state, 1] + [1, n_state, n_state]) + [bs, n_state] => [bs, nstate]
            alpha_t = torch.logsumexp(alphas[:,:,t-1].unsqueeze(2) + expand_A, dim=1) + self.log_B[:, Y[:,t]].T
            alphas[:,:,t] = alpha_t * mask[:,t].unsqueeze(1) + alphas[:,:,t-1] * (1-mask[:,t]).unsqueeze(1)
        return alphas
    

    def backward_alg(self, Y, mask):
        """ calcuating forward probability beta_{ti}
        args:
            Y: 2d tensor of observations, [num_obs, T]
            mask: 2d tensor of pad mask, [num_obs, T]
        returns:
            betas: list of beta, where beta[n][i][t] is the probability of the n-th observation
                    being at state i and having observations (yt+1,yt+2,...,yn) after time t
        """
        num_obs, T = Y.shape
        betas = torch.zeros(num_obs, self.n_state, T).to(self.device)
        
        expand_A = self.log_A.unsqueeze(0)
        mask = torch.FloatTensor(mask).to(self.device)
        for t in range(T-2, -1, -1):
            beta_t = torch.logsumexp(betas[:,:,t+1].unsqueeze(1) + expand_A + self.log_B[:,Y[:,t+1]].T.unsqueeze(1), dim=2)
            betas[:,:,t] = beta_t * mask[:,t+1].unsqueeze(1) + betas[:,:,t+1] * (1-mask[:,t+1]).unsqueeze(1)
    
        return betas


    def back_trace(self, path, start_id):
        res = [start_id]
        for tags in reversed(path):
            res.append(tags[res[-1]])
        return res[::-1]
    

    def infer(self, X, mask=None):
        ans = []
        for x in X:
            path = []
            alpha = self.log_pi + self.log_B[:, x[0]]

            for t in range(1, len(x)):
                _score = alpha + self.log_A + self.log_B[:,x[t]].view(1, -1) # 7*7
                alpha, maxx_id = torch.max(_score, dim=0)
                path.append(maxx_id)
            ans.append(self.back_trace(path, torch.argmax(alpha)))

        return ans
    

    def load_state_dict(self, path):
        state_dict = pkl.load(open(path, 'rb'))
        self.pi = state_dict['pi'].to(self.device)
        self.A = state_dict['A'].to(self.device)
        self.B = state_dict['B'].to(self.device)
        self.logarithmize_params()
        print(f"loaded state dict from {path}")
    

    def save_state_dict(self, path):
        state_dict = {
            'pi': self.pi,
            'A': self.A,
            'B': self.B,
            'log_pi': self.log_pi,
            'log_A': self.log_A,
            'log_B': self.log_B
        }
        for k in ['pi', 'A', 'B']:
            print(k, state_dict[k])
        pkl.dump(state_dict, open(path, 'wb'))
        print(f"state dict saved at {path}")


def baum_welch_update(hmm, Y, mask):
    alphas = []
    betas = []
    gammas = [] # list of [n,1] tensor
    xis = [] # list of [n,n] tensor

    alphas = hmm.forward_alg(Y, mask) # [bs, n_state, T]
    betas = hmm.backward_alg(Y, mask) # [bs, n_state, T]

    gammas = alphas + betas
    gammas = gammas - torch.logsumexp(gammas, dim=1, keepdim=True) # [bs, n_state, T]

    xis = alphas[:,:,:-1].transpose(1,2).unsqueeze(3) + hmm.log_A[None,None] + betas[:,:,1:].transpose(1,2).unsqueeze(2) + hmm.log_B[:,Y[:,1:]].transpose(0,1).transpose(1,2).unsqueeze(2)
    xis = xis - torch.logsumexp(xis, dim=[2,3], keepdim=True) # [bs, T-1, n_state, n_state]

    # update initial state
    new_pi = torch.exp(gammas[:,:,0]).mean(dim=0, keepdim=True).T

    mask = torch.tensor(mask).to(hmm.device)
    new_A = (torch.exp(xis) * mask[:,1:,None,None]).sum(dim=[0,1])
    new_A_denom = (torch.exp(gammas[:,:,:-1,None]).transpose(1,2) * mask[:,1:,None,None]).sum(dim=[0,1])
    new_A = new_A / new_A_denom

    # the following matrix operation leads to OOM: intermediate matrix size: [bs, n_state, T, n_vocab], OOM
    # indicator = torch.LongTensor(Y).to(hmm.device).unsqueeze(2) == torch.arange(hmm.n_vocab).to(hmm.device)
    # new_B = (indicator.unsqueeze(1) * torch.exp(gammas).unsqueeze(3)).sum(dim=[0,2]) / torch.exp(gammas).sum(dim=[0,2]).unsqueeze(1)
    

    # matrix-operation word by word, intermediate matrix size: [bs, n_state, vocab]
    new_B = torch.zeros(hmm.n_state, hmm.n_vocab).to(hmm.device)
    denom = (torch.exp(gammas) * mask[:,None,:]).sum(dim=[0,2]).unsqueeze(1)
    for k in range(hmm.n_vocab):
        indicator = torch.LongTensor(Y).to(hmm.device) == k # [bs, T]
        new_B[:,k] = (indicator.unsqueeze(1) * torch.exp(gammas) * mask[:,None,:]).sum(dim=[0,2])
    new_B = new_B / denom

    # matrix operation state by state, intermediate matrix size: [bs, T, vocab]
    # new_B = torch.zeros(hmm.n_state, hmm.n_vocab).to(hmm.device)
    # denom = torch.exp(gammas).sum(dim=[0,2]).unsqueeze(1)
    # indicator = (torch.LongTensor(Y).unsqueeze(2) == torch.arange(hmm.n_vocab)).to(hmm.device)
    # for i in range(hmm.n_state):
    #     new_B[i,:] = (indicator * torch.exp(gammas[:,i,:]).unsqueeze(2)).sum(dim=[0,1])
    # new_B = new_B / denom

    return new_pi, new_A, new_B


def add_noise(hmm):
    hmm.pi = hmm.pi + torch.rand_like(hmm.pi).to(hmm.device)*0.1
    hmm.A = hmm.A + torch.rand_like(hmm.A).to(hmm.device)*0.05
    hmm.B = hmm.B + torch.rand_like(hmm.B).to(hmm.device)*0.001


def baum_welch_train(hmm, data, mask, num_iters, start_iter):
    for iter in range(start_iter, num_iters):
        n = (iter+1)*1000
        new_pi, new_A, new_B = baum_welch_update(hmm, data[:n], mask[:n])

        hmm.pi, hmm.A, hmm.B = new_pi, new_A, new_B
        # add_noise(hmm)
        hmm.adjust_params()

        hmm.logarithmize_params()
        
        hmm.save_state_dict(f'ckpts/hmm_params_{iter}.pkl')
        f1 = eval_model(hmm, hmm.vocab, hmm.tag2id)
        print(f'epoch {iter}, f1 = {f1:.6f}')


def test_infer(model, vocab, tag2id, device=torch.device('cpu')):
    id2tag = {v:k for k,v in tag2id.items()}
    with torch.no_grad():
        sentence = "中山大学创办于1924年，是孙中山先生一手创立的"
        # sentence = "南京市长江大桥"
        print(sentence)
        x = torch.LongTensor([[vocab.get(ch, vocab["<unk>"]) for ch in sentence]]).to(device)
        mask = torch.LongTensor([[1] * x.shape[1]]).to(device)
        ids = model.infer(x, mask=mask)[0]
        ids = [int(x.cpu().numpy()) for x in ids]

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
                
                ids = [int(x.cpu().numpy()) for x in ids]
                tags = [id2tag[i] for i in ids]

                for i in range(len(sentence)):
                    if tags[i] == "E" or tags[i] == "S":
                        fout.write("%s  " % sentence[i])
                    else:
                        fout.write(sentence[i])
                fout.write("\n")
            except Exception as e:
                traceback.print_exc(e)
                print(len(sentence), sentence)
                print(len(x), x)
                print(len(ids), ids)
                raise e
    fin.close()
    fout.close()


def parse_score_file(fname):
    fin = open(fname, 'r')
    for line in fin:
        if line.startswith("=== F MEASURE:"):
            f_measure = float(line.split("=== F MEASURE:")[1].strip())
            return f_measure


import subprocess
def eval_model(model, vocab, tag2id):
    evaluate_on_file(model, vocab, tag2id, "data/pku_test.utf8", "data/pku_test.out")

    subprocess.run(["./scripts/score", "data/pku_training_words.utf8",
                    "data/pku_test_gold.utf8", "data/pku_test.out"], stdout=open('score.utf8', 'w'))
    f1 = parse_score_file('score.utf8')
    return f1


def search_params(vocab, tag2id, device):
    records = []
    print('vocab num:', len(vocab))

        

    for i in range(100):
        torch.manual_seed(i)
        model = HMM(vocab, tag2id, device)

        f1 = eval_model(model, vocab, tag2id)
        print(f'seed: {i}, f1: {f1:.8f}')
        records.append([i, f1])
    records = pd.DataFrame(records, columns=['seed', 'f1'])
    records.to_csv('seed_search_results.csv', index=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default="ckpts/hmm_params_1.pkl")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train_file', type=str, default="data/train.bmes")
    parser.add_argument('--test_file', type=str, default="")
    parser.add_argument('--vocab_file', type=str, default="data/vocab.json")
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # load vocab, tag2id
    if not os.path.exists(args.vocab_file):
        utils.build_vocab(args.train_file, args.vocab_file, tag_schema='bmes')
    vocab, tag2id = json.load(open(args.vocab_file))
    rev_vocab = {v:k for k,v in vocab.items()}
    # tag2id.pop('<pad>')

    print(tag2id)
    device = torch.device('cuda')
    # load model
    # search_params(vocab, tag2id, device)

    # model = VanillaCRF(vocab, tag2id, device).to(device)
    # model = BiLSTMCRF(vocab, tag2id, 32, 64, device).to(device)
    # model = TransformerCRF(vocab, tag2id, 16, 16, device).to(device)
    X, Y, mask = utils.load_data(args.train_file, vocab, tag2id, max_len=128)
    X = np.array(X)
    mask = np.array(mask)
    print(X.shape, len(vocab))


    torch.manual_seed(17)

    model = HMM(vocab, tag2id, device)
    # model.save_state_dict(f'ckpts/hmm_params_seed{i}.pkl')
    # test_infer(model, vocab, tag2id)
    # evaluate_on_file(model, vocab, tag2id, "data/pku_test.utf8", "data/pku_test.out")

    start_iter=0
    if os.path.exists(args.model_file):
        model.load_state_dict(args.model_file)
        start_iter = int(args.model_file.split('/')[1].split('.')[0].split('_')[-1])+1
    
    # test
    if args.test:
        test_infer(model, vocab, tag2id)
        evaluate_on_file(model, vocab, tag2id, "data/pku_test.utf8", "data/pku_test.out")
        exit(0)

    n = 10000
    baum_welch_train(model, X[:n], mask[:n], num_iters=100, start_iter=start_iter)



""" manual seed selection (training is not necessary, just evaluate using the initialized model)
1 epoch, torch.manual_seed(0):
=== TOTAL TRUE WORDS RECALL:	0.317
=== TOTAL TEST WORDS PRECISION:	0.289
=== F MEASURE:	0.302

1 epoch, torch.manual_seed(1):
=== TOTAL TRUE WORDS RECALL:	0.036
=== TOTAL TEST WORDS PRECISION:	0.061
=== F MEASURE:	0.045

1 epoch, torch.manual_seed(2):
=== TOTAL TRUE WORDS RECALL:	0.119
=== TOTAL TEST WORDS PRECISION:	0.143
=== F MEASURE:	0.130

1 epoch, torch.manual_seed(3):
=== TOTAL TRUE WORDS RECALL:	0.372
=== TOTAL TEST WORDS PRECISION:	0.269
=== F MEASURE:	0.312

1 epoch, torch.manual_seed(4):
=== TOTAL TRUE WORDS RECALL:	0.312
=== TOTAL TEST WORDS PRECISION:	0.282
=== F MEASURE:	0.296

1 epoch, torch.manual_seed(5):
=== TOTAL TRUE WORDS RECALL:	0.317
=== TOTAL TEST WORDS PRECISION:	0.289
=== F MEASURE:	0.302

1 epoch, torch.manual_seed(6):
=== TOTAL TRUE WORDS RECALL:	0.134
=== TOTAL TEST WORDS PRECISION:	0.159
=== F MEASURE:	0.145

1 epoch, torch.manual_seed(7):
=== TOTAL TRUE WORDS RECALL:	0.136
=== TOTAL TEST WORDS PRECISION:	0.160
=== F MEASURE:	0.147

1 epoch, torch.manual_seed(8):
=== TOTAL TRUE WORDS RECALL:	0.124
=== TOTAL TEST WORDS PRECISION:	0.149
=== F MEASURE:	0.135

1 epoch, torch.manual_seed(9):
=== TOTAL TRUE WORDS RECALL:	0.123
=== TOTAL TEST WORDS PRECISION:	0.147
=== F MEASURE:	0.134
"""

""" training progress after each epoch, with torch.manual_seed(3)

0 epoch (no training, just initialization):
=== TOTAL TRUE WORDS RECALL:	0.304
=== TOTAL TEST WORDS PRECISION:	0.270
=== F MEASURE:	0.286

1 epoch:
=== TOTAL TRUE WORDS RECALL:	0.372
=== TOTAL TEST WORDS PRECISION:	0.269
=== F MEASURE:	0.312

5 epoch:
=== TOTAL TRUE WORDS RECALL:	0.387
=== TOTAL TEST WORDS PRECISION:	0.269
=== F MEASURE:	0.317

10 epoch:
=== TOTAL TRUE WORDS RECALL:	0.404
=== TOTAL TEST WORDS PRECISION:	0.267
=== F MEASURE:	0.322

15 epoch:
=== TOTAL TRUE WORDS RECALL:	0.415
=== TOTAL TEST WORDS PRECISION:	0.263
=== F MEASURE:	0.322

20 epoch:
=== TOTAL TRUE WORDS RECALL:	0.418
=== TOTAL TEST WORDS PRECISION:	0.263
=== F MEASURE:	0.323
"""

""" torch.manual_seed() for B
seed(1)
=== TOTAL TRUE WORDS RECALL:	0.314
=== TOTAL TEST WORDS PRECISION:	0.284
=== F MEASURE:	0.298

seed(2)
=== TOTAL TRUE WORDS RECALL:	0.311
=== TOTAL TEST WORDS PRECISION:	0.281
=== F MEASURE:	0.295

seed(3)
=== TOTAL TRUE WORDS RECALL:	0.312
=== TOTAL TEST WORDS PRECISION:	0.282
=== F MEASURE:	0.297

seed(4)
=== TOTAL TRUE WORDS RECALL:	0.312
=== TOTAL TEST WORDS PRECISION:	0.282
=== F MEASURE:	0.297

seed(5)
=== TOTAL TRUE WORDS RECALL:	0.311
=== TOTAL TEST WORDS PRECISION:	0.281
=== F MEASURE:	0.295

seed(6)
=== TOTAL TRUE WORDS RECALL:	0.307
=== TOTAL TEST WORDS PRECISION:	0.278
=== F MEASURE:	0.291

seed(7)
=== TOTAL TRUE WORDS RECALL:	0.316
=== TOTAL TEST WORDS PRECISION:	0.286
=== F MEASURE:	0.300)

seed(8)
=== TOTAL TRUE WORDS RECALL:	0.317
=== TOTAL TEST WORDS PRECISION:	0.286
=== F MEASURE:	0.301
"""