import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pytorch_lightning as pl

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class BlockNoDw(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(BlockNoDw, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, out_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV3_RPiWide(pl.LightningModule):
    def __init__(self, width_mult=1.0, num_classes=2, lr=0.01):
        super().__init__()

        self.lr = lr
        self.features = []

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.features.append(self.conv1)
        self.bn1 = nn.BatchNorm2d(8)
        self.features.append(self.bn1)
        self.hs1 = hswish()
        self.features.append(self.hs1)

        self.bneck = nn.Sequential(
            # kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride
            BlockNoDw(3, 8, 16, 8, nn.ReLU(inplace=True), SeModule(8), 2),
            BlockNoDw(3, 8, 72, 12, nn.ReLU(inplace=True), None, 2),
            BlockNoDw(3, 12, 88, 12, nn.ReLU(inplace=True), None, 1),
            BlockNoDw(5, 12, 96, 20, hswish(), SeModule(20), 2),
            BlockNoDw(5, 20, 240, 20, hswish(), SeModule(20), 1),
            BlockNoDw(5, 20, 240, 20, hswish(), SeModule(20), 1),
            BlockNoDw(5, 20, 120, 24, hswish(), SeModule(24), 1),
            BlockNoDw(5, 24, 144, 24, hswish(), SeModule(24), 1),
            BlockNoDw(5, 24, 288, 48, hswish(), SeModule(48), 2),
            BlockNoDw(5, 48, 576, 48, hswish(), SeModule(48), 1),
            BlockNoDw(5, 48, 576, 48, hswish(), SeModule(48), 1),
#             BlockNoDw(3, 8, 16, 8, nn.ReLU(inplace=True), None, 2),
#             BlockNoDw(3, 8, 72, 12, nn.ReLU(inplace=True), None, 2),
#             BlockNoDw(3, 12, 88, 12, nn.ReLU(inplace=True), None, 1),
#             BlockNoDw(5, 12, 96, 20, hswish(), None, 2),
#             BlockNoDw(5, 20, 240, 20, hswish(), None, 1),
#             BlockNoDw(5, 20, 240, 20, hswish(), None, 1),
#             BlockNoDw(5, 20, 120, 24, hswish(), None, 1),
#             BlockNoDw(5, 24, 144, 24, hswish(), None, 1),
#             BlockNoDw(5, 24, 288, 48, hswish(), None, 2),
#             BlockNoDw(5, 48, 576, 48, hswish(), None, 1),
#             BlockNoDw(5, 48, 576, 48, hswish(), None, 1),
        )

        self.features.extend([block for block in self.bneck])

        self.conv2 = nn.Conv2d(48, 288, kernel_size=1, stride=1, padding=0, bias=False)
        self.features.append(self.conv2)
        self.bn2 = nn.BatchNorm2d(288)
        self.features.append(self.bn2)
        self.hs2 = hswish()
        self.features.append(self.hs2)
        self.linear3 = nn.Linear(288, 960)
        self.bn3 = nn.BatchNorm1d(960)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(960, num_classes)
        self.init_params()

        self.features = nn.Sequential(*self.features)
        self.accuracies = []
        self.losses = []

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_id):
        image, label = train_batch

        image = image.cuda(self.device, non_blocking=True)
        label = label.cuda(self.device, non_blocking=True)
        
        out = self.hs1(self.bn1(self.conv1(image)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)

        loss = F.cross_entropy(out, label)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, val_batch, batch_id):
        image, label = val_batch

        image = image.cuda(self.device, non_blocking=True)
        label = label.cuda(self.device, non_blocking=True)
        
        out = self.hs1(self.bn1(self.conv1(image)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)

        loss = F.cross_entropy(out, label)
        accuracy = torch.mean((torch.argmax(out, 1) == label).type(torch.float32)).item()
        self.log('val_loss', loss)
        self.log('val_acc', accuracy)

        return accuracy
