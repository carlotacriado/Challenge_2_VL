import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN_FC_Classifier(nn.Module):
    def __init__(self):
        super(CNN_FC_Classifier, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) #batch, channels, height, width --> 32, 16, 256, 256
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 32, 16, 128, 128

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) #batch, channels, height, width --> 32, 32, 128, 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 32, 32, 64, 64

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) #batch, channels, height, width --> 32, 64, 64, 64
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 32, 64, 32, 32

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) #batch, channels, height, width --> 32, 128, 32, 32
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 32, 128, 16, 16

        #FC Linear layers
        self.fc1 = nn.Linear(128*16*16, 512)
        self.fc2 = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(-1, 128*16*16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x