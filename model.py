import torch
import torch.nn as nn


# Implementing the network from the architecture drawings
# ** Residual block **
# Conv2d -------------> BatchNorm2d --------> RELU ----------> Conv2d -------------> Add Shortcut -------> RELU
# feature extraction -> stabilize training -> non-linearity -> feature extraction -> stabilize training -> non-linearity
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), 
                nn.BatchNorm2d(out_channels))
            
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut
        out = torch.relu(out_add)
        
        return out

class AudioCNN(nn.Module):
    # classes is set to 50 due to having that many in the dataset
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
        # here I used a for loop to create 3 residual blocks instead of typing them out manually
        # the loop will create 3 residual blocks each having 64 inputs and 64 outputs -> appends the to a list called 'ModuleList'
        # this in tern lets pytorch know that the input is trainable
        self.layer1 = nn.ModuleList(
            [ResidualBlock(64, 64) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i  == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i  == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i  == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)])
        

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)
         
        x = self.avgpool(x)
        # this view is used from pytorch, so view the data (just to reshape the data)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

         


