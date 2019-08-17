"""
Based on
- https://towardsdatascience.com/deep-cars-transfer-learning-with-pytorch-3e7541212e85
- https://colab.research.google.com/github/ivyclare/DeepCars---Transfer-Learning-With-Pytorch/blob/master/Ivy__Deep_Cars_Identifying_Car_Brands.ipynb#scrollTo=V4W-jtIF3LTr

Further investigation:
- https://github.com/qfgaohao/pytorch-ssd
- https://pytorch.org/tutorials/beginner/saving_loading_models.html
- https://github.com/amdegroot/ssd.pytorch
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html?highlight=transfer
- https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
- https://www.datascience.com/blog/transfer-learning-in-pytorch-part-two
- https://www.ritchievink.com/blog/2018/04/12/transfer-learning-with-pytorch-assessing-road-safety-with-computer-vision/
- https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce
- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD/src
- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

On Bias
- https://becominghuman.ai/bias-variables-in-neural-networks-c36596ff0bab
- https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/

Education
- https://cs231n.github.io
- https://github.com/yosinski/deep-visualization-toolbox
- https://arxiv.org/abs/1506.06579
"""

import os
import torch
from torchvision import datasets,transforms,models

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_dir = os.path.join(os.path.dirname(__file__), '../images') #TODO rename to train
test_dir = os.path.join(os.path.dirname(__file__), '../test')
label_path = os.path.join(os.path.dirname(__file__), '../annotations/vott-csv-export/ToyNet-export.csv')
save_dir = os.path.join(os.path.dirname(__file__), '../model')

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# Tansform with data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        # transforms.RandomResizedCrop(299),  #size for inception architecture
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        # transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
}

batch_size=32
dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])

# splitting our data
valid_size = int(0.1 * len(dataset))
train_size = len(dataset) - valid_size
dataset_sizes = {'train': train_size, 'valid': valid_size}

# now we get our datasets
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

# Loading datasets into dataloader
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True),
    'valid': torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = False),
}

label_df = pd.read_csv(label_path, names=["label"])
