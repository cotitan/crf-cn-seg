""" apply HMM on Chinese Word Segmentation task,
    implement both in a sequential manner and in a batch manner (todo)
"""

import torch
from torch import nn
import os
import utils
import argparse
import json
import pickle as pkl
import traceback
from tqdm import tqdm


class HMM(nn.Module):
    def __init__(self, vocab, tag2id, device):
        super(HMM, self).__init__()
        torch.manual_seed(42)
        self.device = device
        self.n_vocab = len(vocab)
        self.n_state = len(tag2id)
        self.vocab = vocab
        self.tag2id = tag2id
        self.init_params(self.n_vocab, self.n_state)

    def init_params(self, n_vocab, n_state):
        tag2id = self.tag2id
        # BMES
        self.pi = torch.rand(n_state, 1).to(self.device)
        self.pi[tag2id['M'],0] = 0 # prior, initial states is B or S
        self.pi[tag2id['E'],0] = 0
        self.pi = self.pi / self.pi.sum(axis=0)

        self.A = torch.rand(n_state, n_state).to(self.device)
        self.A[tag2id['B'],tag2id['B']] = 0
        self.A[tag2id['B'],tag2id['S']] = 0
        self.A[tag2id['M'],tag2id['B']] = 0
        self.A[tag2id['M'],tag2id['S']] = 0
        self.A[tag2id['E'],tag2id['M']] = 0
        self.A[tag2id['E'],tag2id['E']] = 0
        self.A[tag2id['S'],tag2id['M']] = 0
        self.A[tag2id['S'],tag2id['E']] = 0
        self.A = self.A / self.A.sum(axis=1, keepdim=True)

        self.B = torch.rand(n_state, n_vocab).to(self.device)
        self.B = self.B / self.B.sum(axis=1, keepdim=True)

        self.logarithmize_params()

        print("pi:")
        print(self.pi)
        print("A:")
        print(self.A)

        print("log_pi:")
        print(self.log_pi)
        print("log_A:")
        print(self.log_A)


    def logarithmize_params(self):
        self.log_pi = torch.log(self.pi)
        self.log_pi = torch.masked_fill(self.log_pi, self.log_pi==float('-inf'), -1000)
        self.log_A = torch.log(self.A)
        self.log_A = torch.masked_fill(self.log_A, self.log_A==float('-inf'), -1000)
        self.log_B = torch.log(self.B)
    

    """
    logsumexp usage: to implement a matmul in log-space
    x = [e,e**2,e**3], log_x = [1,2,3]
    y = [e**3,e**2,e], log_y = [3,2,1]
    x.dot(y) = e**4 * 3
    log_(xdoty) = logsumexp(log_x + log_y = [4,4,4]) = logsum([e**4, e**4, e**4])
                = log(e**4 *3) = 4 + log(3)
    """

    def forward_alg(self, obs):
        """ calcuating forward probability alpha_{ti}
        args:
            Y: list of Observations (could be tensor or list)
        returns:
            alphas: list of alpha, where alpha[t][i] is the probability of
                    being at state i and having observation y_t at time t,
                    and having observations (y1,y2...yt-1) before time t
        """
        # pi * B [4,1] * [4,N] => [4,1]
        alpha = []
        alpha.append(self.log_pi + self.log_B[:,obs[0]].unsqueeze(1))

        for t in range(1, len(obs)):
            # alpha[-1].T @ A @ B[idx]  [1,4] @ [4,4] * [4] => [4,1]
            alpha_t = torch.logsumexp(alpha[-1] + self.log_A, dim=0, keepdim=True).T + self.log_B[:, obs[t]].view(-1,1)
            alpha.append(alpha_t)
        return alpha
    

    def backward_alg(self, obs):
        """ calcuating forward probability beta_{ti}
        args:
            obs: list of Observations (could be tensor or list)
        returns:
            betas: list of beta, where beta[t][i] is the probability of
                    being at state i and having observation y_t at time t,
                    and having observations (yt+1,yt+2,...,yn) after time t
        """
        beta = []
        beta.append(torch.zeros(self.n_state, 1).to(self.device))
        for t in range(len(obs)-2, -1, -1):
            # beta_{t+1}
            beta_t = torch.logsumexp(beta[-1].T + self.log_A + self.log_B[:,obs[t+1]].view(1,-1),
                                        dim=1, keepdim=True)
            beta.append(beta_t)
    
        return beta[::-1]


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
        print(state_dict)
        pkl.dump(state_dict, open(path, 'wb'))
        print(f"state dict saved at {path}")


def baum_welch_update(hmm, Y):
    alphas = []
    betas = []
    gammas = [] # list of [n,1] tensor
    xis = [] # list of [n,n] tensor

    for r,obs in tqdm(enumerate(Y), total=len(Y)):
        alpha = hmm.forward_alg(obs)
        beta = hmm.backward_alg(obs)

        alphas.append(alpha)
        betas.append(beta)

        gamma = []
        for t in range(len(obs)):
            gamma.append(alpha[t] + beta[t] - torch.logsumexp(alpha[t] + beta[t], dim=0))
            # print("gamma:", alpha[t].shape, beta[t].shape, torch.logsumexp(alpha[t] + beta[t], dim=0).shape)
        gammas.append(gamma)

        xi = []
        for t in range(len(obs)-1):
            xi_t = alpha[t] + hmm.log_A + beta[t+1].T + hmm.log_B[:,obs[t+1]].view(1,-1)
            denom = torch.logsumexp(xi_t, dim=[0,1])
            xi.append(xi_t - denom)
        xis.append(xi)

    # update initial state
    new_pi = torch.cat([torch.exp(g[0]) for g in gammas], dim=1).mean(dim=1, keepdim=True)

    # update transition matrix
    new_A = torch.zeros(hmm.n_state, hmm.n_state).to(device)
    new_A_denom = torch.zeros(hmm.n_state, 1).to(device)
    for r,obs in enumerate(Y):
        for t in range(len(obs)-1):
            new_A += torch.exp(xis[r][t])
            new_A_denom += torch.exp(gammas[r][t])
    new_A = new_A / new_A_denom

    # update emission matrix
    new_B = torch.zeros(hmm.n_state, hmm.n_vocab).to(device)
    new_B_denom = torch.zeros(hmm.n_state, 1).to(device)
    for r,obs in enumerate(Y):
        indicator = torch.arange(hmm.n_vocab).unsqueeze(0).expand(hmm.n_state, -1).to(hmm.device)
        for t in range(len(obs)):
            new_B += torch.exp(gammas[r][t]) * (indicator == obs[t]).float()
            new_B_denom += torch.exp(gammas[r][t])
    new_B = new_B / new_B_denom

    return new_pi, new_A, new_B


def baum_welch_train(hmm, data, num_iters, start_iter):
    for iter in range(start_iter,num_iters):
        new_pi, new_A, new_B = baum_welch_update(hmm, data)

        hmm.pi, hmm.A, hmm.B = new_pi, new_A, new_B
        hmm.logarithmize_params()
        print(hmm.pi.shape, hmm.A.shape, hmm.B.shape)
        print(hmm.log_pi.shape, hmm.log_A.shape, hmm.log_B.shape)
        
        hmm.save_state_dict(f'data/hmm_params_{iter}.pkl')


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default="data/hmm_params_1.pkl")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train_file', type=str, default="data/train.bmes")
    parser.add_argument('--test_file', type=str, default="")
    parser.add_argument('--vocab_file', type=str, default="data/vocab.json")
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # load vocab, tag2id
    n_examples = 20000
    if not os.path.exists(args.vocab_file):
        utils.build_vocab(args.train_file, args.vocab_file, n_examples)
    vocab, tag2id = json.load(open(args.vocab_file))
    rev_vocab = {v:k for k,v in vocab.items()}
    # tag2id.pop('<pad>')

    print(tag2id)
    # load model
    device = torch.device('cuda')
    model = HMM(vocab, tag2id, device)
    for k, p in model.named_parameters():
        print(k, p.shape)

    start_iter=0
    if os.path.exists(args.model_file):
        state_dict = pkl.load(open(args.model_file, 'rb'))
        print(state_dict)
        for k in state_dict:
            print(k, state_dict[k].shape)
        model.load_state_dict(args.model_file)
        start_iter = int(args.model_file.split('/')[1].split('.')[0].split('_')[-1])+1
    
    # test
    if args.test:
        test_infer(model, vocab, tag2id)
        evaluate_on_file(model, vocab, tag2id, "data/pku_test.utf8", "data/pku_test.out")
        exit(0)

    X, Y = utils.load_data(args.train_file, vocab, tag2id, n_examples)
    baum_welch_train(model, X[:1000], num_iters=10, start_iter=start_iter)



"""
before train:
=== TOTAL TRUE WORDS RECALL:	0.046
=== TOTAL TEST WORDS PRECISION:	0.073
=== F MEASURE:	0.056

after 1 epoch:
=== TOTAL TRUE WORDS RECALL:	0.036
=== TOTAL TEST WORDS PRECISION:	0.066
=== F MEASURE:	0.046

after 2 epochs:
=== TOTAL TRUE WORDS RECALL:	0.036
=== TOTAL TEST WORDS PRECISION:	0.066
=== F MEASURE:	0.047

after 3 epochs:
=== TOTAL TRUE WORDS RECALL:	0.037
=== TOTAL TEST WORDS PRECISION:	0.067
=== F MEASURE:	0.048

after 8 epochs:
=== TOTAL TRUE WORDS RECALL:	0.050
=== TOTAL TEST WORDS PRECISION:	0.088
=== F MEASURE:	0.064
"""