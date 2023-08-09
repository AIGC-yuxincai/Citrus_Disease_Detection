import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class se_block(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b, c])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])
        outputs = x * inputs
        return outputs

class HsBranch_block(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(HsBranch_block, self).__init__()
        self.fc = nn.Linear(in_features=in_channel, out_features=224, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=224, out_features=224 // ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=224 // ratio, out_features=224, bias=False)
        self.fc3 = nn.Linear(224,in_channel)
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(input_size=in_channel,hidden_size=4,num_layers=1, batch_first=True,bidirectional=True)

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b, c])

        x1 = self.fc(x)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        x1 = self.sigmoid(x1)
        x = x * x1
        x, _ = self.lstm(x)
        return x

class Ourmodel(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 channels = 44,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(Ourmodel, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group
        self.conv1 = nn.Conv2d(channels, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=4,out_features=1)
        self.spectral_branch = HsBranch_block(channels,4)

        self.se_block_first = se_block(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)

        self.se_block_final = se_block(256)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(264, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, h, w = x.shape
        x_spectral = self.spectral_branch(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.se_block_final(x)
        x = self.avgpool(x)
        x = x.view([b,256])
        x = torch.cat([x_spectral, x], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def Our_model(num_classes=1000, channels=44,include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return Ourmodel(BasicBlock, [1, 2, 2], num_classes=num_classes, channels=channels,include_top=include_top)

if __name__ == '__main__':

    model = Our_model(4,44)

    from torchsummaryX import summary
    # 打印模型结构
    dummy_input = torch.randn(1, 44, 224, 224)
    summary(model, dummy_input)
