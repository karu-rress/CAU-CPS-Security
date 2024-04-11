# torchvision 관련 에러 메시지를 제거합니다.
import warnings
warnings.filterwarnings("ignore", category=UserWarning,
    message="Failed to load image Python extension")

from models import *
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import argparse
import torchvision


model_dict = {
    'simple': SimpleNet,
    'lenet5': LeNet5,
    'resnet50': ResNet,
    'mobilenet': MobileNet, #  torchvision.models.mobilenet_v3_small,  << 잘못
}
dataset_dict = {
    'cifar10': {
        'loader': datasets.CIFAR10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.247, 0.243, 0.261),
    },
    'mnist': {
        'loader': datasets.MNIST,
        'mean': (0.1307,),
        'std': (0.3081,),
    },
}


def get_model(device=torch.device('cuda'),
              _args=argparse.Namespace()):
    if _args.model in model_dict:
        model = model_dict[_args.model](_args.dataset).to(device)
    else:
        raise ValueError(f'model {_args.model} is not supported')

    return model


def get_data(_args,
             is_train=True,
             ):
    if _args.dataset in dataset_dict.keys():
        temp_data = dataset_dict[_args.dataset]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(temp_data['mean'], temp_data['std']),
        ])
        dataset = temp_data['loader'](str(Path(__file__).parent / 'data'),
                                      train=is_train, download=True, transform=transform)
    else:
        raise ValueError(f'dataset {_args.dataset} is not supported')

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

            pred = output.argmax(dim=1, keepdim=True)  # 확률이 가장 높은 class를 예측값으로 선택
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / data_len
    accuracy = 100. * correct / data_len

    return avg_loss, accuracy


def main(_args):
    torch.set_num_threads(12)
    
    # Environments configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model selection
    model = get_model(device, _args)
    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=_args.lr)

    # Model training
    train_loader = get_data(_args, is_train=True)
    for epoch in tqdm(range(_args.epochs), total=_args.epochs, unit='epoch', desc='Training'):
        train_loss, train_acc = run(model, train_loader, loss_fn, optimizer, device=device, is_train=True)
        print(f'Epoch {epoch + 1}/{_args.epochs}: Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')

    # Model test
    test_loader = get_data(_args, is_train=False)
    test_loss, test_acc = run(model, test_loader, loss_fn, device=device, is_train=False)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')


# 수정 전 파일이라 굳이 required=True로 바꾸진 않았습니다.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple', required=False,
                        help='model name')
    parser.add_argument('--dataset', type=str, default='cifar10', required=False,
                        help='dataset name')
    parser.add_argument('--epochs', type=int, default=10, required=False,
                        help='# of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, required=False,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, required=False,
                        help='learning rate (default: 0.01)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Called with cfgs:')
    for arg, value in vars(args).items():
        print(f"\t{arg}: {value}")
    main(args)
