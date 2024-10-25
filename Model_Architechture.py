import torch.nn as nn
import torch.nn.functional as F

class BrainTumorModel(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorModel, self).__init__()

        # Convolutional Block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm1_2 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm2_2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm3_3 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.batchnorm4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batchnorm4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batchnorm4_3 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batchnorm5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batchnorm5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batchnorm5_3 = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.batchnorm_fc1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.batchnorm_fc2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Convolutional Block 1
        x = F.relu(self.batchnorm1_1(self.conv1_1(x)))
        x = F.relu(self.batchnorm1_2(self.conv1_2(x)))
        x = self.maxpool1(x)

        # Convolutional Block 2
        x = F.relu(self.batchnorm2_1(self.conv2_1(x)))
        x = F.relu(self.batchnorm2_2(self.conv2_2(x)))
        x = self.maxpool2(x)

        # Convolutional Block 3
        x = F.relu(self.batchnorm3_1(self.conv3_1(x)))
        x = F.relu(self.batchnorm3_2(self.conv3_2(x)))
        x = F.relu(self.batchnorm3_3(self.conv3_3(x)))
        x = self.maxpool3(x)

        # Convolutional Block 4
        x = F.relu(self.batchnorm4_1(self.conv4_1(x)))
        x = F.relu(self.batchnorm4_2(self.conv4_2(x)))
        x = F.relu(self.batchnorm4_3(self.conv4_3(x)))
        x = self.maxpool4(x)

        # Convolutional Block 5
        x = F.relu(self.batchnorm5_1(self.conv5_1(x)))
        x = F.relu(self.batchnorm5_2(self.conv5_2(x)))
        x = F.relu(self.batchnorm5_3(self.conv5_3(x)))
        x = self.maxpool5(x)

        # Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.batchnorm_fc1(self.fc1(x)))
        x = F.relu(self.batchnorm_fc2(self.fc2(x)))
        x = self.fc3(x)

        return x