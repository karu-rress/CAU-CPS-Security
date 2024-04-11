import argparse
import torch
import torchvision

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm


def data_load(args, is_train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalization for CIFAR10
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    dataset = datasets.CIFAR10(str(Path(__file__).parent/'data'), train=is_train, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return data_loader


def train_model(model, data_loader, loss_fn, optimizer, epochs=10,
                device=torch.device("cuda")):
    # 학습 모드 설정
    model.train()

    pbar = tqdm(range(epochs), total=epochs, unit='epoch')
    for epoch in pbar:
        train_loss, correct = 0, 0
        for data, target in data_loader:
            # data와 target을 모델이 실행되고 있는 장치로 이동
            data, target = data.to(device), target.to(device)

            # 기울기 버퍼를 0으로 설정 (새로운 최적화 단계를 시작하기 전에 필요)
            optimizer.zero_grad()

            # 데이터에 대한 모델의 예측 수행
            output = model(data)

            # 손실 계산
            loss = loss_fn(output, target)
            train_loss += loss.item()

            # backpropagation: 현재 손실 값(loss)에 대한 gradient를 계산
            loss.backward()

            # 계산된 gradient를 기반하여 모델의 weight를 실제로 업데이트
            optimizer.step()

            # 정확도 계산
            pred = output.argmax(dim=1, keepdim=True)  # 확률이 가장 높은 class를 예측값으로 선택
            correct += pred.eq(target.view_as(pred)).sum().item()

        # epoch마다 평균 손실과 정확도 출력
        train_loss /= len(data_loader.dataset)
        accuracy = 100. * correct / len(data_loader.dataset)
        pbar.write(f'Epoch {epoch + 1}: Average loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%')


def test_model(model, test_loader, loss_fn, device=torch.device("cuda")):
    model.eval()  # 모델을 평가 모드로 설정

    test_loss, correct = 0, 0
    with torch.no_grad():  # 기울기 계산을 비활성화
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, required=True,
                        help='# of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, required=True,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, required=True,
                        help='learning rate (default: 0.01)')

    return parser.parse_args()


def main(args):
    # Environments configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection
    pretrained_weight = torchvision.models.MobileNet_V2_Weights.DEFAULT
    model = torchvision.models.mobilenet_v2(weights=pretrained_weight).to(device)

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Train data load
    train_dataloader = data_load(args=args, is_train=True)
    # Model training
    train_model(model, train_dataloader, loss_fn, optimizer, epochs=args.epochs, device=device)

    # Test data load
    test_dataloader = data_load(args=args, is_train=False)
    # Model testing
    test_model(model, test_dataloader, loss_fn, device=device)


if __name__ == '__main__':
    args = parse_args()
    main(args)