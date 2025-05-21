"""
Special loss functions for our experiments.
"""

import torch
import torch.nn as nn

class RegularizedCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.1, p=2.0, device='cpu'):
        super(RegularizedCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.p = p
        self.device = device
        print('RegularizedCrossEntropyLoss initialized with alpha:', alpha, 'p:', p)

    def forward(self, output, target, params=[], output_reg=False):

        # print()
        # print(output, target)
        # loss = nn.CrossEntropyLoss()(output, target)
        loss = nn.NLLLoss(reduction="mean")(output, target)

        reg_loss = torch.tensor(0.0).to(self.device)

        # print(loss, reg_loss)

        for param in params:
            reg_loss += torch.norm(param, p=self.p)

        if output_reg:
            if self.alpha == 0:
                return loss, reg_loss
            return loss + self.alpha * reg_loss, reg_loss
    
        
        return loss + self.alpha * reg_loss
    

class MeanConstrainedActivationsCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.1, use_groups=False, p=2.0, device='cpu'):
        super(MeanConstrainedActivationsCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.p = p
        self.device = device
        self.use_groups = use_groups

    def forward(self, output, target, activations=None, params=[], output_reg=False):

        if activations is None:
            print('Activations are None')
            msa = torch.tensor(0.0).to(self.device)
        elif self.use_groups:
            # define groups by having same target
            groups = target.unique()
            msa = 0
            n = 0
            for group in groups:
                group_activations = activations[target == group]
                msa += self.msa(group_activations)
                n += 1
            msa /= n
                
        else:
            msa = self.msa(activations)

        loss = nn.CrossEntropyLoss()(output, target)
        
        if output_reg:
            return loss + self.alpha * msa, msa
        
        return loss + self.alpha * msa
    
    def msa(self, activations):
        msa = torch.mean(activations, dim=0)
        diffs = activations - msa
        squares = torch.pow(torch.norm(diffs, p=self.p), 2)
        msa = torch.mean(squares)
        return msa
    

class PairwiseDifferencesRegularizedCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.1, p=2.0, sample=100, device='cpu'):
        super(PairwiseDifferencesRegularizedCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.p = p
        self.device = device
        self.sample = sample

    def forward(self, output, target, activations=None, params=[], output_reg=False):

        if activations is None:
            print('Activations are None')
            msa = 0
        else:
            # sample random pairs with same target
            pairs = []
            for i in range(len(target)):
                for j in range(i+1, len(target)):
                    if len(pairs) >= self.sample:
                            break
                    if target[i] == target[j]:
                        pairs.append((i, j))
            
            # calculate pairwise differences
            diffs = []
            for i, j in pairs:
                diffs.append(activations[i] - activations[j])

            # calculate norm of pairwise differences
            loss = torch.pow(torch.norm(diffs, p=self.p), 2)
            reg = torch.sum(loss) 

        loss = nn.CrossEntropyLoss()(output, target)

        if output_reg:
            return loss + self.alpha * reg, reg
    
        return loss + self.alpha * reg