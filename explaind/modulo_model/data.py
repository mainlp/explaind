"""
Data handling for the modulo model.

Parts of this are taken from the pytorch transformer tutorial and modified for toy datasets.

https://github.com/pytorch/examples/blob/main/word_language_model/data.py
"""

import torch
import numpy as np


class ModulusDataGenerator:

    def __init__(self,
                 num_samples=-1,
                 val_num_samples=200,
                 test_num_samples=200,
                 P=113, 
                 power=1, 
                 no_val_leak=True,
                 add_arithmetic_tokens=True):
        self.num_samples = num_samples
        self.P = P
        self.power = power
        self.val_num_samples = val_num_samples
        self.test_num_samples = test_num_samples
        self.no_val_leak = no_val_leak
        self.add_arithmetic_tokens = add_arithmetic_tokens

    def generate_data(self):
        # get all possible combinations of a and be \in \{0, 1, ..., P-1\}
        # avoid duplicates by only considering pairs (a, b) where a <= b
        pairs = []
        for a in range(self.P):
            for b in range(a, self.P):
                pairs.append((b, a))
        a, b = zip(*pairs)
        a, b = np.array(a), np.array(b)
        
        # calculate the target value
        target = np.power(a + b, self.power) % self.P
        
        # sample val and test sets
        test_indices = np.random.choice(len(a), self.test_num_samples, replace=False)
        val_indices = np.random.choice(list(set(range(len(a))) - set(test_indices)), self.val_num_samples, replace=False)
        if self.num_samples == -1:
            train_indices = list(set(range(len(a))) - set(test_indices) - set(val_indices))
        else:
            train_indices = np.random.choice(list(set(range(len(a))) - set(test_indices) - set(val_indices)), self.num_samples, replace=False)
        a_test, b_test = a[test_indices], b[test_indices]
        target_test = target[test_indices]
        # plus and equality signs
        plus_test = np.ones(self.test_num_samples) * self.P
        eq_test = np.ones(self.test_num_samples) * (self.P + 1)

        a_val, b_val = a[val_indices], b[val_indices]
        target_val = target[val_indices]
        # plus and equality signs
        plus_val = np.ones(self.val_num_samples) * self.P
        eq_val = np.ones(self.val_num_samples) * (self.P + 1)

        a, b = a[train_indices], b[train_indices]
        target = target[train_indices]
        # plus and equality signs
        plus = np.ones(self.num_samples) * self.P
        eq = np.ones(self.num_samples) * (self.P + 1)

        if self.add_arithmetic_tokens:
            return ((list(zip(a, plus, b, eq)), target),        
                    (list(zip(a_val, plus_val, b_val, eq_val)), target_val),
                    (list(zip(a_test, plus_test, b_test, eq_test)), target_test))

        return ((list(zip(a, b)), target),
                (list(zip(a_val, b_val)), target_val),
                (list(zip(a_test, b_test)), target_test))


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, train, val, test):
        self.dictionary = Dictionary()
        self.train = self.tokenize(train)
        self.val = self.tokenize(val)
        self.test = self.tokenize(test)

    def tokenize(self, dataset):
        """Tokenizes a text file."""

        for sample in dataset:
            words = list(sample) + ['<eos>']
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        dataset_ids = []
        for sample in dataset:
            print(sample)
            words = list(sample) + ['<eos>']
            ids = []
            for word in words:
                ids.append(self.dictionary.word2idx[word])
            dataset_ids.append(torch.tensor(ids).type(torch.int64))
            print(ids)
        
        dataset_ids = torch.cat(dataset_ids)

        return dataset_ids


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.int)
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)