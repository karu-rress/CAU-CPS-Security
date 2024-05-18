import argparse

import torch
from torch.utils.data import DataLoader

import torchvision.models as models
from torchvision import datasets, transforms

from pathlib import Path
from tqdm.auto import tqdm

from optimizer import SGD, Momentum, RMSprop, Adam


def get_data(_args,
             is_train=True,
             ):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    dataset = datasets.CIFAR10(str(Path('./data')), train=is_train, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=_args.batch_size, shuffle=True, num_workers=4)

    return data_loader


def run(model,
        dataloader,
        loss_fn,
        optimizer=None,
        is_train=True,
        device=torch.device('cuda'),
        ):
    total_loss, correct = 0.0, 0
    data_len = len(dataloader.dataset)
    model.train() if is_train else model.eval()

    with torch.set_grad_enabled(is_train):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / data_len
    accuracy = 100. * correct / data_len

    return avg_loss, accuracy


def main(_args):
    # Environments configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model selection
    pretrained_weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=pretrained_weights).to(device)

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizers = {
        'sgd': SGD(model.parameters(), lr=_args.lr),
        'momentum': Momentum(model.parameters(), lr=_args.lr, momentum=0.9),
        'rmsprop': RMSprop(model.parameters(), lr=_args.lr, alpha=0.99),
        'adam': Adam(model.parameters(), lr=_args.lr)
    }

    optimizer = optimizers[_args.optimizer]

    # Model training
    train_loader = get_data(_args, is_train=True)
    for epoch in tqdm(range(_args.epochs), total=_args.epochs, unit='epoch', desc='Training'):
        train_loss, train_acc = run(model, train_loader, loss_fn, optimizer, device=device, is_train=True)
        print(f'Epoch {epoch + 1}/{_args.epochs}: Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')

    # Model test
    test_loader = get_data(_args, is_train=False)
    test_loss, test_acc = run(model, test_loader, loss_fn, device=device, is_train=False)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'momentum', 'rmsprop', 'adam'],
                        required=True, help='optimizer to use (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.01, required=False,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10, required=True,
                        help='# of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, required=True,
                        help='input batch size for training (default: 64)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    for arg, value in vars(args).items():
        print(f"\t{arg}: {value}")
    main(args)
