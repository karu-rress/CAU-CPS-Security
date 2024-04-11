import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

# cifar10은 컬러 이미지이므로 3개 채널(RGB),
# MNIST는 흑백 이미지이므로 1개 채널을 가집니다.
channels: dict[str, int] = { 'cifar10': 3, 'mnist': 1 }

class SimpleNet(nn.Module):
    def __init__(self, dataset: str):
        super(SimpleNet, self).__init__()
        
        # 데이터셋 이름을 호출자로부터 받아, 저장합니다.
        # 해당 데이터셋에 맞는 channel 값을 가져옵니다.
        self.dataset: str = dataset
        self.channels: int = channels[self.dataset]
        
        # 데이터셋에 따라 channel 값을 알맞게 적용합니다.
        self.conv1 = nn.Conv2d(self.channels, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        
        # 마찬가지로, 완전연결층의 개수도 알맞게 조절합니다.
        self.fc1 = nn.Linear(32 * ((8 if dataset == 'cifar10' else 7)**2), 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))   # 3*32*32 -> 16*32*32   # 1*28*28 -> 16*28*28
        x = torch.max_pool2d(x, 2)      # 16*32*32 -> 16*16*16  # 16*28*28 -> 16*14*14
        x = torch.relu(self.conv2(x))   # 16*16*16 -> 32*16*16  # 16*14*14 -> 32*14*14
        x = torch.max_pool2d(x, 2)      # 32*16*16 -> 32*8*8    # -> 32*7*7
        x = torch.flatten(x, start_dim=1, end_dim=-1)  # 32*8*8 -> 2048 # 32*7*7 -> 1568
        x = torch.relu(self.fc1(x))     # 2048 -> 128
        x = self.fc2(x)                 # 128 -> 10
        return x


class LeNet5(nn.Module):
    def __init__(self, dataset):
        super(LeNet5, self).__init__()
        
        # 데이터셋 이름을 호출자로부터 받아, 저장합니다.
        # 해당 데이터셋에 맞는 channel 값을 가져옵니다.
        self.dataset = dataset
        self.channels = channels[dataset]
        
        # Sequential에서는 순서를 맞춰줘야 하므로 OrderedDict 사용
        self.base = nn.Sequential(
            OrderedDict(
                # 데이터셋에 따라 channel 값을 알맞게 적용합니다.
                conv1=nn.Conv2d(self.channels, 6, kernel_size=5, stride=1, padding=0),  # 3*32*32 -> 6*28*28 # 1*28*28 -> 6*24*24
                activation1=nn.ReLU(),
                pool1=nn.AvgPool2d(2),  # 6*28*28 -> 6*14*14 # 6*24*24 -> 6*12*12
                
                conv2=nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # 6*14*14 -> 16*10*10 # 6*12*12 -> 16*8*8
                activation2=nn.ReLU(),
                pool2=nn.AvgPool2d(2),  # 16*10*10 -> 16*5*5 # 16*8*8 -> 16*4*4
                
                flatten=nn.Flatten(start_dim=1),  # 16*5*5 -> 400
                fc1=nn.Linear(400 if self.dataset == "cifar10" else 256, 120),  # 400/256 -> 120
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),  # 120 -> 84
                activation4=nn.ReLU(),
                fc3=nn.Linear(84, 10),  # 84 -> 10
            )
        )

    def forward(self, x):
        return self.base(x)


class ResNet(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        
        self.dataset = dataset
        self.channels = channels[dataset]
        
        pretrained_weights = models.ResNet50_Weights.DEFAULT
        resnet: models.ResNet = models.resnet50(weights=pretrained_weights)
        
        # MNIST 데이터셋을 사용한다면, 채널 값 수정
        if self.dataset == 'mnist':
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, 10)
        self.base.fc = nn.Identity()

    def forward(self, x):
        return self.classifier(self.base(x))


class MobileNet(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        
        self.dataset = dataset
        self.channels = channels[dataset]
        
        pretrained_weights = models.MobileNet_V3_Small_Weights.DEFAULT
        mobilenet: models.MobileNetV3 = models.mobilenet_v3_small(weights=pretrained_weights)
        
        # MNIST 데이터셋을 사용한다면, 채널 값 수정
        if self.dataset == 'mnist':
            mobilenet.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        
        self.base = mobilenet
        self.classifier = nn.Linear(self.base.classifier[-1].in_features, 10)
        self.base.classifier[-1] = nn.Identity()

    def forward(self, x):
        return self.classifier(self.base(x))
