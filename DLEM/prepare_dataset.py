import torch
import random
import numpy as np

from torch.utils import data
from transformers import AutoTokenizer
from .augment import Augmenter

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}


def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class SSLDataset(data.Dataset):
    """
    Dataset for pre-training using 
    Partially from: https://github.com/megagonlabs/sudowoodo/blob/main/selfsl/bt_dataset.py
    """
    def __init__(self, X, max_len=128, size=1000, lm='distilbert', da='all'):
        self.tokenizer = get_tokenizer(lm)
        self.instances= []
        self.max_len = max_len
        self.size = size

        '''
        if size is not None:
            if size > len(self.instances):
                N = size // len(self.instances) + 1
                self.instances = (self.instances * N)[:size]
            else:
                self.instances = random.sample(self.instances, size)
        '''

        self.X_entityA = X[:, 0]
        self.X_entityB = X[:, 1]
        if size is None:
            self.instances = np.concatenate((self.X_entityA, self.X_entityB))

        self.da = da    # default is random
        self.augmenter = Augmenter()
        self.ground_truth = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.instances)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the 1st entity
            List of int: token ID's of the 2nd entity
        """
        if self.da == 'cutoff':
            # A = B = self.instances[idx]
            A = self.instances[idx]
            # combine with the deletion operator
            B = self.augmenter.augment_sent(A, "del") 
        else:
            A = self.instances[idx]
            B = self.augmenter.augment_sent(A, self.da)

        # left
        yA = self.tokenizer.encode(text=A,
                                   max_length=self.max_len,
                                   truncation=True)
        yB = self.tokenizer.encode(text=B,
                                   max_length=self.max_len,
                                   truncation=True)
        return yA, yB

    @staticmethod
    def pad(batch):
        """
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
        """
        yA, yB = zip(*batch)

        max_len = max([len(x) for x in yA + yB])
        yA = [xi + [0]*(max_len - len(xi)) for xi in yA]
        yB = [xi + [0]*(max_len - len(xi)) for xi in yB]

        return torch.LongTensor(yA), \
               torch.LongTensor(yB)
    


class EMDataset(data.Dataset):
    """
    Data set for EM
    """

    def __init__(self, X, y, max_len=128, lm='distilbert', size=500):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = X
        self.labels = y
        self.max_len = max_len
        
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the 1st entity
            List of int: token ID's of the 2nd entity
            List of int: token ID's of the two entities combined
            int: the label of the pair (0: unmatch, 1: match)
        """
        entity_a = self.pairs[idx][0]
        entity_b = self.pairs[idx][1]

        # entity A
        x1 = self.tokenizer.encode(text=entity_a,
                                    max_length=self.max_len,
                                    truncation=True)
        # entity B
        x2 = self.tokenizer.encode(text=entity_b,
                                    max_length=self.max_len,
                                    truncation=True)
        # entity A + entity B
        x12 = self.tokenizer.encode(text=entity_a,
                                    text_pair=entity_b,
                                    max_length=self.max_len,
                                    truncation=True)
        return x1, x2, x12, self.labels[idx]
    
    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: x12 of shape (batch_size, seq_len').
                        Elements of x12 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        x1, x2, x12, y = zip(*batch)

        maxlen = max([len(x) for x in x1+x2])
        x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
        x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]

        maxlen = max([len(x) for x in x12])
        x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]

        return torch.LongTensor(x1), \
                torch.LongTensor(x2), \
                torch.LongTensor(x12), \
                torch.LongTensor(y)
