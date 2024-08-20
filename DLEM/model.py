import numpy as np
import random
import torch
from torch import nn
from transformers import AutoModel

lm_mp = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased'
}

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class EMModel(nn.Module):
    def __init__(self, device='cuda', lm='distilbert'):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)
        self.device = device

        hidden_size = self.bert.config.hidden_size
        # projector as proposed in SimCLR
        proj_out_size = 768
        self.projector = nn.Linear(hidden_size, proj_out_size)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(proj_out_size, affine=False)

        # a fully connected layer for fine tuning
        self.fc = nn.Linear(proj_out_size * 2, 2)
        
    def forward(self, flag, y1, y2, y12, da=None, cutoff_ratio=0.05):
        """ 
        Adapted from https://github.com/megagonlabs/sudowoodo/blob/main/selfsl/barlow_twins_simclr.py
        """
        if flag in [0, 1]:
            # pre-training
            # encode
            batch_size = len(y1)
            y1 = y1.to(self.device) # original
            y2 = y2.to(self.device) # augment
            if da == 'cutoff':
                seq_len = y2.size()[1]
                y1_word_embeds = self.bert.embeddings.word_embeddings(y1)
                y2_word_embeds = self.bert.embeddings.word_embeddings(y2)

                # modify the position embeddings of y2
                position_ids = torch.LongTensor([list(range(seq_len))]).to(self.device)
                pos_embeds = self.bert.embeddings.position_embeddings(position_ids)

                # sample again
                l = random.randint(1, int(seq_len * cutoff_ratio)+1)
                s = random.randint(0, seq_len - l - 1)
                y2_word_embeds[:, s:s+l, :] -= pos_embeds[:, s:s+l, :]

                # merge y1 and y2
                y_embeds = torch.cat((y1_word_embeds, y2_word_embeds))
                z = self.bert(inputs_embeds=y_embeds)[0][:, 0, :]
            else:
                # cat y1 and y2 for faster training
                y = torch.cat((y1, y2))
                z = self.bert(y)[0][:, 0, :]
            z = self.projector(z)
            if flag == 0:
                return z
            elif flag == 1:
                # barlow twins
                z1 = z[:batch_size]
                z2 = z[batch_size:]

                # empirical cross-correlation matrix
                c = (self.bn(z1).T @ self.bn(z2)) / (len(z1))

                # use --scale-loss to multiply the loss by a constant factor
                on_diag = ((torch.diagonal(c) - 1) ** 2).sum() * 1.0/256
                # off_diag = off_diagonal(c).pow_(2).sum().mul(1.0/256)
                off_diag = (off_diagonal(c) ** 2).sum() * 1.0/256
                loss = on_diag + 3.9e-3 * off_diag
                return loss
        elif flag == 2:
            # fine tuning
            x1 = y1
            x2 = y2
            x12 = y12

            x1 = x1.to(self.device) # (batch_size, seq_len)
            x2 = x2.to(self.device) # (batch_size, seq_len)
            x12 = x12.to(self.device) # (batch_size, seq_len)
            # left+right
            enc_pair = self.projector(self.bert(x12)[0][:, 0, :]) # (batch_size, emb_size)
            batch_size = len(x1)
            # left and right
            enc = self.projector(self.bert(torch.cat((x1, x2)))[0][:, 0, :])
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)
            logits = self.fc(torch.cat((enc_pair, (enc1 - enc2).abs()), dim=1)) # .squeeze() # .sigmoid()
            embeddings = [enc_pair, enc1, enc2]
            return logits, embeddings
