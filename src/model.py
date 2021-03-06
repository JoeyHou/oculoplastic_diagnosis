import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets
# import torchvision.transforms as transforms
# from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
# import torch.optim as optim
# from datetime import date


# define the CNN architecture
class DiagnoisisNet(nn.Module):
    def __init__(self, config):
        super(DiagnoisisNet, self).__init__()
        self.original_w, self.original_h = config['original_size']
        self.num_pooling = config['num_pooling']
        self.final_w = int(self.original_w / (2 ** self.num_pooling))
        self.final_h = int(self.original_h / (2 ** self.num_pooling))
        self.chann1 = config['chann1']
        self.chann2 = config['chann2']
        self.chann3 = config['chann3']
        self.mid_dim = config['mid_dim']
        self.class_num = config['class_num']

        self.conv1 = nn.Conv2d(1, self.chann1, 3, padding = 1)
        self.conv2 = nn.Conv2d(self.chann1, self.chann2, 3, padding = 1)
        self.conv3 = nn.Conv2d(self.chann2, self.chann3, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.chann3 * self.final_w * self.final_h, self.mid_dim)
        self.fc2 = nn.Linear(self.mid_dim, self.class_num)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)
        x = x.view(-1, self.chann3 * self.final_w * self.final_h)
        # x = x.reshape(self.chann3 * self.final_w * self.final_h)
        # print(x.shape)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
