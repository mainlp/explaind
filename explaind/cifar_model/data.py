"""
Dataset loader for CIFAR as used in the TRAK paper and published here:

    https://github.com/MadryLab/trak/blob/main/examples/cifar_quickstart.ipynb
"""
import torchvision
import torch
import warnings
warnings.filterwarnings('ignore')


def get_dataloader(batch_size=256, num_workers=8, split='train', shuffle=False, augment=True, type="cifar10", drop_last=True):
    if augment:
        transforms = torchvision.transforms.Compose(
                        [torchvision.transforms.RandomHorizontalFlip(),
                         torchvision.transforms.RandomAffine(0),
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])
    else:
        transforms = torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])
        
    is_train = (split == 'train')
    if type == "cifar10":
    
        dataset = torchvision.datasets.CIFAR10(root='/tmp/cifar/',
                                                download=True,
                                                train=is_train,
                                                transform=transforms)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                                shuffle=shuffle,
                                                batch_size=batch_size,
                                                num_workers=num_workers, drop_last=drop_last)
    elif type == "cifar2":
        dataset = torchvision.datasets.CIFAR10(root='/tmp/cifar/',
                                                download=True,
                                                train=is_train,
                                                transform=transforms)
        # remove classes that are not cats (id 3) or dogs (id 5)
        dataset.data = dataset.data[(torch.Tensor(dataset.targets) == 3) | (torch.Tensor(dataset.targets) == 5), :, :, :]
        dataset.targets = [0 if x == 3 else 1 for x in dataset.targets if x == 3 or x == 5]

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                                shuffle=shuffle,
                                                batch_size=batch_size,
                                                num_workers=num_workers, drop_last=drop_last)

    
    return loader, dataset