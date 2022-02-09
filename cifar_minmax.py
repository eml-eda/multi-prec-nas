import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

PATH='/space/risso/multi_prec_exp/cifar10'

transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

trainset = torchvision.datasets.CIFAR10(root= PATH, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
            #trainset, batch_size=len(trainset), shuffle=True,
            trainset, batch_size=10, shuffle=True,
            num_workers=2, pin_memory=True, sampler=None)

min_dict = dict()
max_dict = dict()
for i, (images, label) in enumerate(train_loader):
    min_dict['ch_0'] = torch.min(images[:, 0, :, :])
    min_dict['ch_1'] = torch.min(images[:, 1, :, :])
    min_dict['ch_2'] = torch.min(images[:, 2, :, :])

    max_dict['ch_0'] = torch.max(images[:, 0, :, :])
    max_dict['ch_1'] = torch.max(images[:, 1, :, :])
    max_dict['ch_2'] = torch.max(images[:, 2, :, :])

for key, val in min_dict.items():
    print(f"{key} min: {val}")

for key, val in max_dict.items():
    print(f"{key} max: {val}")

'''
"clip_val": {
            "ch_0": [
                -2.429065704345703,
                2.5140879154205322
            ],
            "ch_1": [
                -2.418254852294922,
                2.596790313720703
            ],
            "ch_2": [
                -2.22139310836792,
                2.7537312507629395
            ]
        }, 
'''