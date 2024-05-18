import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as models
from torchvision import datasets, transforms

from collections import OrderedDict
from pathlib import Path


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.classes = 10
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, self.classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.classes = 10
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
                activation1=nn.ReLU(),
                pool1=nn.AvgPool2d(2),
                conv2=nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                activation2=nn.ReLU(),
                pool2=nn.AvgPool2d(2),
                flatten=nn.Flatten(start_dim=1),
                fc1=nn.Linear(400, 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
                activation4=nn.ReLU(),
                fc3=nn.Linear(84, self.classes),
            )
        )

    def forward(self, x):
        return self.base(x)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classes = 10
        pretrained_weights = models.ResNet50_Weights.DEFAULT
        resnet: models.ResNet = models.resnet50(weights=pretrained_weights)
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, self.classes)
        self.base.fc = nn.Identity()

    def forward(self, x):
        return self.classifier(self.base(x))


def get_model(device=torch.device('cuda'),
              _args=argparse.Namespace(),
              ):
    if _args.model not in model_dict.keys():
        raise ValueError(f'model {_args.model} is not supported')

    return model_dict[_args.model]().to(device)


def get_data(_args,
             is_train=False,
             ):
    if _args.dataset not in dataset_dict.keys():
        raise ValueError(f'dataset {_args.dataset} is not supported')

    temp_data = dataset_dict[_args.dataset]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(temp_data['mean'], temp_data['std']),
    ])

    dataset = temp_data['loader'](str(Path('./data')), train=is_train, download=True, transform=transform)

    data_loader = DataLoader(dataset, batch_size=_args.batch_size, shuffle=True, num_workers=4)

    return data_loader


model_dict = {
    'simple': SimpleNet,
    'lenet5': LeNet5,
    'resnet': ResNet,
}
dataset_dict = {
    'cifar10': {
        'loader': datasets.CIFAR10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.247, 0.243, 0.261),
        'classes': 10,
    },
    'mnist': {
        'loader': datasets.MNIST,
        'mean': (0.1307,),
        'std': (0.3081,),
        'classes': 10,
    },
}
