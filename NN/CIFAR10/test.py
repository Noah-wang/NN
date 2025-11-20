import torch
import os
import cv2
import torchvision.transforms as transforms
from torch.serialization import add_safe_globals
from net import CIFAR10Net

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

use_gpu = torch.backends.mps.is_available()
add_safe_globals([CIFAR10Net])
model = torch.load(
    "./model/model_300.pth",
    map_location="mps" if use_gpu else "cpu",
    weights_only=False,
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

folder_path = "./test_image/"

files = os.listdir(folder_path)

print("开始测试图片分类效果...")
imageh_files = [os.path.join(folder_path, file) for file in files]

for img in imageh_files:
    image = cv2.imread(img)
    cv2.imshow("input image", image)
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = transform(image)
    image = torch.reshape(image, (1, 3, 32, 32))
    image = image.to("mps" if use_gpu else "cpu")
    output = model(image)

    value, index = torch.max(output, dim=1)
    pre_val = classes[index.item()]
    print("图片 {} 预测类别: {}".format(img, pre_val))
    cv2.waitKey(0)
