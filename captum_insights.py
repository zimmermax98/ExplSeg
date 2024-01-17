import os
import sys
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# use locally modified version of captum
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "ExplSeg"))
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

dataset = "cifar10"

def get_classes(dataloader):
    if dataset == "cifar10":
        classes = [
        "Plane",
        "Car",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
        ]
        return classes
    elif dataset == "imagenet":
        classname_file = os.path.join(os.path.dirname(dataloader.dataset.root), "imagenet1000_clsidx_to_labels.json")
        with open(classname_file) as f:
            folder_label_map = json.load(f)
        
        class_labels = {i: folder_label_map[folder] for i, folder in enumerate(dataloader.dataset.classes)}
        return list(class_labels.values())


def get_pretrained_model():
    if dataset == "cifar10":
        class Net(nn.Module):
            def __init__(self) -> None:
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool1 = nn.MaxPool2d(2, 2)
                self.pool2 = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
                self.relu1 = nn.ReLU()
                self.relu2 = nn.ReLU()
                self.relu3 = nn.ReLU()
                self.relu4 = nn.ReLU()

            def forward(self, x):
                x = self.pool1(self.relu1(self.conv1(x)))
                x = self.pool2(self.relu2(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = self.relu3(self.fc1(x))
                x = self.relu4(self.fc2(x))
                x = self.fc3(x)
                return x

        net = Net().to("cuda")
        pt_path = "/visinf/home/vimb03/ExplSeg/captum/captum/insights/attr_vis/models/cifar_torchvision.pt"
        net.load_state_dict(torch.load(pt_path))
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset == "imagenet":
        weights = torchvision.models.VGG11_Weights.IMAGENET1K_V1
        net = torchvision.models.vgg11(weights=weights).to("cuda")
        t = weights.transforms()
    
    net.eval()

    return net, t


def baseline_func(input):
    return input * 0


def get_dataloader(transforms):
    if dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root="/fastdata/dlcv_group_f/datasets", train=False, download=False, transform=transforms
        )
    elif dataset == "imagenet":
        dataset = torchvision.datasets.ImageFolder("/fastdata/MT_ExplSeg/datasets/imagenet/val", transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    return dataloader


def formatted_data_iter(dataloader, model):
    dataloader = iter(dataloader)
    while True:
        images, labels = next(dataloader)
        images.requires_grad = True
        images = images.to("cuda")
        labels = labels.to("cuda")
        outputs = model(images).argmax(dim=1)
        
        yield Batch(inputs=images[outputs == labels], labels=labels[outputs == labels])

def get_unnormalize(transforms):
    if dataset == "cifar10":
        mean = -torch.Tensor(transforms.transforms[1].mean)
        std = 1 / torch.Tensor(transforms.transforms[1].std)
    elif dataset == "imagenet":
        mean = - torch.Tensor(transforms.mean)
        std = 1 / torch.Tensor(transforms.std)

    unnormalize = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[ 0., 0., 0. ], std=std),
                                                  torchvision.transforms.Normalize(mean=mean, std=[ 1., 1., 1. ])])
    return unnormalize


def main():
    model, transforms = get_pretrained_model()
    dataloader = get_dataloader(transforms)
    unnormalize = get_unnormalize(transforms)
    visualizer = AttributionVisualizer(
        models=[model],
        score_func=lambda o: torch.nn.functional.softmax(o, 1),
        classes=get_classes(dataloader),
        features=[
            ImageFeature(
                "Photo",
                baseline_transforms=[baseline_func],
                input_transforms=[],
                visualization_transform=unnormalize
            )
        ],
        dataset=formatted_data_iter(dataloader, model),
    )

    visualizer.serve(debug=True)


if __name__ == "__main__":
    main()