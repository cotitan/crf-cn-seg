import math
import torch
from torch import nn


class CRF(nn.Module):
    """
    Pytorch implementation of CRF, for sequence labeling

    Attributes
    ----------
    transition: torch.tensor, the transition function, shape of (n_tags, n_tags)
    start_trans: torch.tensor, shape of (n_tags,), transition function at the begging
    end_trans:  torch.tensor, shape of (n_tags,), transition function at the end

    Methods
    -------
    forward(X, Y, mask)
        calculate `log p(Y|X)`
    
    """

    def __init__(self, num_labels, pad_idx, device) -> None:
        """transition: (ntags, ntags)
             pad  t1    t2    t3   </s>
        pad   0  -inf  -inf  -inf  -inf # 不能从pad转移到其他位置，也不可以从其他位置转移到pad，可以从</s>转移到pad
         t1 -inf  0     0     0     0
         t2 -inf  0     0     0     0
         t3 -inf  0     0     0     0
        <s> -inf  0     0     0         # 可以转移到任何位置，一般不可以转移到pad和</s>，因为序列长度一般>1
                                    ↑   # 可以从其他位置转移到</s>，不能从pad转移到</s>
         
        """
        super(CRF, self).__init__()
        self.device = device
        self.transition = torch.randn(num_labels, num_labels)
        self.start_trans = torch.randn(num_labels)
        self.end_trans = torch.randn(num_labels)

        if pad_idx is not None:
            self.start_trans[pad_idx] = -10000.0
            self.transition[pad_idx, :] = -10000.0
            self.transition[:, pad_idx] = -10000.0
            self.transition[pad_idx, pad_idx] = 0.0
            self.end_trans[pad_idx] = -10000.0
        
        self.transition = nn.Parameter(self.transition)
        self.start_trans = nn.Parameter(self.start_trans)
        self.end_trans = nn.Parameter(self.end_trans)

        
    def forward(self, X, Y, mask):
        """
        calculate `log p(Y|X)`, Y is the batch of labelling, X is the batch of observation state vectors

        Parameters:
        x (torch.FloatTensor): in shape (batch_size, seq_len, n_tags)
        y (torch.LongTensor): in shape (batch_size, seq_len)
        mask (torch.BoolTensor): in shape (batch_size, seq_len), for padding mask

        Returns:
        torch.FloatTensor: `log p(y|x)`
        """

        log_p = self._compute_numerator(X, Y, mask)
        log_Z = self._compute_denominator(X, mask)
        # log(p/Z) = log_P - log_Z
        return log_p - log_Z
    
    
    def _compute_numerator(self, X, Y, mask):
        """ """
        bs, seqlen, ntag = X.shape
        rg_bs = torch.arange(bs).to(self.device)

        score = self.start_trans[Y[:,0]] + X[rg_bs, 0 ,Y[:,0]]
        for t in range(1, seqlen):
            score += (self.transition[Y[:,t-1],Y[:,t]] + X[rg_bs,t,Y[:,t]]) * mask[:,t]
        each_len = mask.sum(dim=1).int()
        score += self.end_trans[Y[rg_bs, each_len-1]] * mask[rg_bs, each_len-1]
        return score


    def _compute_denominator(self, X, mask):
        """
        """
        bs, seqlen, ntag = X.shape
        score = self.start_trans + X[:,0] # bs, ntag
        trans = self.transition.unsqueeze(0) # 1, ntag, ntag

        for t in range(1, seqlen):
            last_score = score.unsqueeze(2)# bs, ntag, 1
            score_t = last_score + trans + X[:,t].unsqueeze(1) # bs, ntag, ntag
            score_t = torch.logsumexp(score_t, dim=1) # bs, ntag

            mask_t = mask[:,t].unsqueeze(1) # bs, 1
            score = torch.where(mask_t.bool(), score_t, score)
        score = score + self.end_trans
        return torch.logsumexp(score, dim=1)


    def back_trace(self, path, start_id, seqlen):

        res = [start_id]
        for tags in reversed(path[:seqlen]):
            res.append(tags[res[-1]].item())
        return res[::-1]
    

    def viterbi_decode(self, X, mask):
        bs, seqlen, ntag = X.shape
        score = self.start_trans.data + X[:,0] # bs, ntag
        trans = self.transition.unsqueeze(0) # 1, ntag, ntag

        paths, scores = [], []
        for t in range(1, seqlen):
            last_score = score.unsqueeze(2)# bs, ntag, 1
            score_t = last_score + trans + X[:,t].unsqueeze(1) # bs, ntag, ntag

            max_score, path = score_t.max(dim=1)
            paths.append(path)
            scores.append(max_score)
            score = max_score

        score = score + self.end_trans.data # bs, ntag

        paths.append(path) # seqlen, bs

        # get tags seperately
        return [self.back_trace([p[i] for p in paths],
                                torch.argmax(score[i]).item(),
                                mask[i].sum()) for i in range(bs)]


class VanillaCRF(nn.Module):
    def __init__(self, vocab, tag2id, device) -> None:
        super(VanillaCRF, self).__init__()
        self.num_tags = len(tag2id)
        self.emit_score = nn.Embedding(len(vocab), self.num_tags, padding_idx=0)
        self.crf = CRF(self.num_tags, pad_idx=0, device=device)
        self.device = device

        
    def forward(self, X, Y, mask):
        """ calculate `-log p(Y|X)`, which is the loss 
        
        Parameters:
        X: torch.LongTensor, (batch, seqlen)
        Y: torch.LongTensor, (batch, seqlen)
        mask: torch.BoolTensor (batch, seqlen), padding mask, 0 for padding
        """
        emit_score = self.emit_score(X)
        log_likelihood = self.crf.forward(emit_score, Y, mask)
        return -log_likelihood.sum() / mask.sum()
    
    
    def infer(self, X, mask):
        emit_score = self.emit_score(X)
        return self.crf.viterbi_decode(emit_score, mask)


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab, tag2id, emb_dim, hid_dim, device) -> None:
        super(BiLSTMCRF, self).__init__()
        self.device = device
        
        self.word_embeds = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hid_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        
        self.num_tags = len(tag2id)
        self.hidden2tag = nn.Linear(hid_dim, self.num_tags)
        self.crf = CRF(self.num_tags, pad_idx=0, device=device)

        
    def forward(self, X, Y, mask):
        """ calculate `-log p(Y|X)`, which is the loss 
        
        Parameters:
        X: torch.LongTensor, (batch, seqlen)
        Y: torch.LongTensor, (batch, seqlen)
        mask: torch.BoolTensor (batch, seqlen), padding mask, 0 for padding
        """
        embeds = self.word_embeds(X)
        encoder_out, _ = self.encoder(embeds)
        tag_feats = self.hidden2tag(encoder_out)
        log_likelihood = self.crf.forward(tag_feats, Y, mask)
        return -log_likelihood.sum() / mask.sum()
    
    
    def infer(self, X, mask):
        embeds = self.word_embeds(X)
        encoder_out, _ = self.encoder(embeds)
        tag_feats = self.hidden2tag(encoder_out)
        return self.crf.viterbi_decode(tag_feats, mask)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)
    
class TransformerCRF(nn.Module):
    def __init__(self, vocab, tag2id, emb_dim, hid_dim, device) -> None:
        super(TransformerCRF, self).__init__()
        self.device = device
        
        self.word_embeds = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(hid_dim, nhead=4, dim_feedforward=hid_dim*4,
                                                   batch_first=True, dropout=0.0, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.num_tags = len(tag2id)
        self.hidden2tag = nn.Linear(hid_dim, self.num_tags)
        self.crf = CRF(self.num_tags, pad_idx=0, device=device)
        self.pe = PositionalEncoding(hid_dim, dropout=0.0)

        
    def forward(self, X, Y, mask):
        """ calculate `-log p(Y|X)`, which is the loss 
        
        Parameters:
        X: torch.LongTensor, (batch, seqlen)
        Y: torch.LongTensor, (batch, seqlen)
        mask: torch.BoolTensor (batch, seqlen), padding mask, 0 for padding
        """
        embeds = self.pe(self.word_embeds(X))
        encoder_out = self.encoder(embeds)
        tag_feats = self.hidden2tag(encoder_out)
        log_likelihood = self.crf.forward(tag_feats, Y, mask)
        return -log_likelihood.sum() / mask.sum()
    
    
    def infer(self, X, mask):
        embeds = self.pe(self.word_embeds(X))
        encoder_out = self.encoder(embeds)
        tag_feats = self.hidden2tag(encoder_out)
        return self.crf.viterbi_decode(tag_feats, mask)

