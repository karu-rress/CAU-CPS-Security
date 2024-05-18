from torchvision.utils import save_image
from tqdm.auto import tqdm

from utils import *


def fgsm(model, loss_fn, data, target, atk_conf):
    epsilon = atk_conf['epsilon']

    data.requires_grad = True

    output = model(data)
    model.zero_grad()
    loss = loss_fn(output, target)
    loss.backward()

    # perturbation을 계산
    perturbation = epsilon * data.grad.sign()
    # perturbation이 추가된 이미지 픽셀의 값을 [0,1] 범위로 조정
    perturbed_data = torch.clamp(data + perturbation, 0, 1)

    return perturbed_data


def pgd(model, loss_fn, data, target, atk_conf):
    epsilon = atk_conf['epsilon']
    alpha = atk_conf['alpha']
    num_iter = atk_conf['num_iter']

    origin_data = data.clone()
    for i in range(num_iter):
        data.requires_grad = True

        output = model(data)
        model.zero_grad()
        loss = loss_fn(output, target)
        loss.backward()

        # perturbation을 추가함
        perturbation = data + alpha * data.grad.sign()
        # 원본 이미지로부터 epsilon 범위를 벗어나지 않도록 조정
        eta = torch.clamp(perturbation - origin_data, min=-epsilon, max=epsilon)
        data = torch.clamp(origin_data + eta, min=0, max=1).detach_()

    return data


def adv(model,
        dataloader,
        loss_fn,
        atk_type='fgsm',
        device=torch.device('cuda'),
        ):
    if atk_type not in ['fgsm', 'pgd']:
        raise ValueError(f'atk_type {atk_type} is not supported')

    atk_func = {
        'fgsm': fgsm,
        'pgd': pgd,
    }
    atk_conf = {
        'epsilon': 32./255.,
        'alpha': 16./255.,
        'num_iter': 10,
    }
    total_loss, correct = 0.0, 0
    data_len = len(dataloader.dataset)
    model.eval()

    origin_examples = list()
    adv_examples = list()
    pbar = tqdm(dataloader, desc='Generate the adversarial examples')

    for data, target in pbar:

        data, target = data.to(device), target.to(device)

        if len(origin_examples) < 10:
            origin_img = data.detach().cpu()
            origin_examples.append(origin_img)

        perturbed_image = atk_func[atk_type](model, loss_fn, data, target, atk_conf)

        output = model(perturbed_image)
        loss = loss_fn(output, target)

        if len(adv_examples) < 10:
            adv_img = perturbed_image.detach().cpu()
            adv_examples.append(adv_img)

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # 확률이 가장 높은 class를 예측값으로 선택
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / data_len
    accuracy = 100. * correct / data_len

    origin_path = Path(f"./origin_examples")
    origin_path.mkdir(exist_ok=True)
    adv_path = Path(f"./adv_examples")
    adv_path.mkdir(exist_ok=True)

    for idx, (origin_img, adv_img) in enumerate(zip(origin_examples, adv_examples)):
        save_image(origin_img, Path(origin_path / f'{idx}.png'))
        save_image(adv_img, Path(adv_path / f'{idx}.png'))

    return avg_loss, accuracy


def main(_args):
    # Environments configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model selection
    model = get_model(device, _args)
    model.classes = dataset_dict[_args.dataset]['classes']
    model.load_state_dict(torch.load(Path(_args.pretrained), map_location=torch.device('cpu')))

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Adversarial attack
    adv_loader = get_data(_args, is_train=False)
    adv_loss, adv_acc = adv(model, adv_loader, loss_fn, atk_type=_args.atk_type, device=device)
    print(f'Adversarial attack Loss: {adv_loss:.4f}, Accuracy: {adv_acc:.2f}%')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--atk-type', type=str, default='pgd', required=True,
                        help='attack type')
    parser.add_argument('--model', type=str, default='resnet', required=True,
                        help='model name')
    parser.add_argument('--dataset', type=str, default='cifar10', required=True,
                        help='dataset name')
    parser.add_argument('--pretrained', type=str, default='./checkpoints/resnet_cifar10.pt', required=True,
                        help='model name')
    parser.add_argument('--batch-size', type=int, default=8, required=False,
                        help='input batch size for training (default: 8)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    for arg, value in vars(args).items():
        print(f"\t{arg}: {value}")
    main(args)
