from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from CIFAR10.model import CIFAR10Net
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = Path(__file__).resolve().parent
writer = SummaryWriter(str(BASE_DIR / "logs"))

def main():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    checkpoint_dir = BASE_DIR / "model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("正在下载CIFAR10数据集...")
    train_dataset = datasets.CIFAR10(
        str(BASE_DIR / "./data"), train=True, transform=transform, download=True
    )

    print("正在下载CIFAR10测试数据集...")
    test_dataset = datasets.CIFAR10(
        str(BASE_DIR / "./data"), train=False, transform=transform, download=True
    )

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = CIFAR10Net()

    use_gpu = torch.mps.is_available()
    if use_gpu:
        model = model.to("mps")
        print("使用GPU进行训练")
    else:
        print("使用CPU进行训练")

    epochs = 300

    # 定义损失函数和优化器
    lossFn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))

        # 损失变量
        train_total_loss = 0.0
        test_total_loss = 0.0

        # 准确率
        train_total_acc = 0.0
        test_total_acc = 0.0

        # 开始训练
        model.train()
        for data in train_loader:
            inputs, labels = data

            if use_gpu:
                inputs, labels = inputs.to("mps"), labels.to("mps")
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = lossFn(outputs, labels)
            loss.backward()
            # 更新参数
            optimizer.step()

            _, index = torch.max(outputs, 1)
            acc = torch.sum(index == labels).item()

            train_total_loss += loss.item()
            train_total_acc += acc

        model.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data

                if use_gpu:
                    inputs, labels = inputs.to("mps"), labels.to("mps")
                outputs = model(inputs)

                loss = lossFn(outputs, labels)

                _, index = torch.max(outputs, 1)
                acc = torch.sum(index == labels).item()

                test_total_loss += loss.item()
                test_total_acc += acc

        train_loss_avg = train_total_loss / len(train_loader)
        test_loss_avg = test_total_loss / len(test_loader)
        train_acc_avg = 100.0 * train_total_acc / train_dataset_size
        test_acc_avg = 100.0 * test_total_acc / test_dataset_size

        print(
            "训练集上的损失: {:.3f}，准确率: {:.3f}%".format(
                train_loss_avg,
                train_acc_avg,
            )
        )
        print(
            "测试集上的损失: {:.3f}，准确率: {:.3f}%".format(
                test_loss_avg,
                test_acc_avg,
            )
        )

        writer.add_scalar("Loss/train", train_loss_avg, epoch)
        writer.add_scalar("ACC/train", train_acc_avg, epoch)
        writer.add_scalar("Loss/test", test_loss_avg, epoch)
        writer.add_scalar("ACC/test", test_acc_avg, epoch)

        if (epoch + 1) % 50 == 0:
            torch.save(model, str(checkpoint_dir / f"model_{epoch + 1}.pth"))


if __name__ == "__main__":
    main()
