from torch.autograd import Variable
import argparse
from datetime import datetime
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage.io as io
from thop import profile
from thop import clever_format
from skimage.morphology import dilation, disk
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PolypDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.image_root = image_root
        self.gt_root = gt_root
        self.samples = [name for name in os.listdir(image_root) if name[0] != "."]
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            #            A.RandomRotate90(p=0.2),
            ToTensorV2()
        ])

        self.color1, self.color2 = [], []
        for name in self.samples:
            if name[:-10].isdigit():
                self.color1.append(name)
            else:
                self.color2.append(name)

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.image_root + '/' + name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        name2 = self.color1[idx % len(self.color1)] if np.random.rand() < 0.7 else self.color2[idx % len(self.color2)]
        image2 = cv2.imread(self.image_root + '/' + name2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

        mean, std = image.mean(axis=(0, 1), keepdims=True), image.std(axis=(0, 1), keepdims=True)
        mean2, std2 = image2.mean(axis=(0, 1), keepdims=True), image2.std(axis=(0, 1), keepdims=True)
        image = np.uint8((image - mean) / std * std2 + mean2)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        mask = cv2.imread(self.gt_root + '/' + name, cv2.IMREAD_GRAYSCALE) / 255.0
        pair = self.transform(image=image, mask=mask)

        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True,
               augmentation=False):
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model.urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model.urls['res2net101_v1b_26w_4s']))
    return model


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model_state = torch.load('Snapshots/Res2net/res2net50.pth')
        model.load_state_dict(model_state)
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model.urls['res2net101_v1b_26w_4s']))
    return model


def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model.urls['res2net152_v1b_26w_4s']))
    return model


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0):]))


def CalParams(model, input_tensor):
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel, n_class):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x2_2 = self.conv_upsample5(self.upsample(x2_2))

        # x2_2_rs = -1 * (torch.sigmoid(x2_2)) + 1
        # x2_2 = x2_2 * x2_2_rs

        x3_1_rs = -1 * (torch.sigmoid(x3_1)) + 1

        x3_2 = torch.cat((x3_1_rs, torch.sigmoid(x2_2)), 1)  # 反向信息
        # x3_2 = torch.cat((x3_2, x2_2_rs), 1)
        # x3_2 = torch.cat((x3_2, x3_1_rs), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3, ],
                 channel_split=[1, 3, 4, ],
                 ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims - self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                 ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.ReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class DeConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, output_padding, dilation=(1, 1), groups=1, bn_acti=False,
                 bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.ConvTranspose2d(nIn, nOut, kernel_size=kSize,
                                       stride=stride, padding=padding, output_padding=output_padding,
                                       dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


def get_sobel(in_chan, out_chan):
    '''
    filter_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]).astype(np.float32)
    filter_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(2048, 256)
        self.reduce4 = Conv1x1(256, 256)
        self.block = nn.Sequential(
            ConvBNR(512, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x1, p2):
        x1 = self.reduce1(x1)
        p2 = self.reduce4(p2)
        x1 = F.interpolate(x1, scale_factor=8, mode='bilinear', align_corners=False)
        out = torch.cat((x1, p2), dim=1)
        out = self.block(out)

        return out


class Encoder(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, n_class=1):
        super(Encoder, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        # ---- high-level features ----
        # x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        return x1, x2, x3, x4


class Conv2(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv2, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class aggregation2(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel, n_class):
        super(aggregation2, self).__init__()
        self.relu = nn.ReLU(True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, n_class, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(x1) * x2
        x3_1 = self.conv_upsample2(x1) * self.conv_upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)
        x2_2 = self.conv_upsample5(x2_2)

        # x2_2_rs = -1 * (torch.sigmoid(x2_2)) + 1
        # x2_2 = x2_2 * x2_2_rs

        x3_1_rs = -1 * (torch.sigmoid(x3_1)) + 1

        x3_2 = torch.cat((x3_1_rs, torch.sigmoid(x2_2)), 1)  # 反向信息
        # x3_2 = torch.cat((x3_2, x2_2_rs), 1)
        # x3_2 = torch.cat((x3_2, x3_1_rs), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        self.agg1 = aggregation2(ch_int, ch_out)

        # bi-linear modelling for both
        self.W_g = Conv2(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv2(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv2(ch_int, ch_int, 3, bn=True, relu=True)


        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)
        # fuse = self.residual(torch.cat([g, x, bp], 1))
        fuse = self.agg1(W_g, W_x, bp)

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class Decoder1(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, n_class=1, drop_rate=0.2):
        super(Decoder1, self).__init__()
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        self.MOG_agg = MultiOrderGatedAggregation(embed_dims=32)
        # self.MOG_agg_2 = MultiOrderGatedAggregation(embed_dims=32)
        # self.MOG_agg_3 = MultiOrderGatedAggregation(embed_dims=256)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel, n_class)
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(256, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, n_class, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)

        self.up_c1 = BiFusion_block(ch_1=2048, ch_2=2048, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c2 = BiFusion_block(ch_1=1024, ch_2=1024, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)
        self.up_c3 = BiFusion_block(ch_1=512, ch_2=512, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate / 2)

    def forward(self, edge, x2, x3, x4):
        # ---- Decoder 1 ----
        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x2_rfb = self.MOG_agg(x2_rfb) + x2_rfb


        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x3_rfb = self.MOG_agg(x3_rfb) + x3_rfb

        x4_rfb = self.rfb4_1(x4)  # channel -> 32
        x4_rfb = self.MOG_agg(x4_rfb) + x4_rfb
        # edge = edge.repeat(1, 8, 1, 1)

        agg = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_4 = F.interpolate(agg, scale_factor=8,
                                      mode='bilinear')  # NOTES: Sup-1 (bs, 1, 22, 22) -> (bs, 1, 44, 44)
        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(agg, scale_factor=0.25, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = self.up_c1(x.expand(-1, 2048, -1, -1), x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        x = self.ra4_conv5(x)
        x = x + crop_4
        lateral_map_3 = F.interpolate(x, scale_factor=32,
                                      mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = self.up_c2(x.expand(-1, 1024, -1, -1), x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        x = self.ra3_conv4(x)
        x = x + crop_3
        lateral_map_2 = F.interpolate(x, scale_factor=16,
                                      mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = self.up_c3(x.expand(-1, 512, -1, -1), x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        x = self.ra2_conv4(x)
        x = x + crop_2
        lateral_map_1 = F.interpolate(x, scale_factor=8,
                                      mode='bilinear')  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_4


class Decoder2(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, n_class=1):
        super(Decoder2, self).__init__()
        # --- sobel decoder---
        self.sobel_x1, self.sobel_y1 = get_sobel(256, 256)
        self.sobel_x2, self.sobel_y2 = get_sobel(512, 1)
        self.sobel_x3, self.sobel_y3 = get_sobel(1024, 1)
        self.sobel_x4, self.sobel_y4 = get_sobel(2048, 2048)
        self.eam = EAM()

    def forward(self, x1, x4):
        s1 = run_sobel(self.sobel_x1, self.sobel_y1, x1)
        s4 = run_sobel(self.sobel_x4, self.sobel_y4, x4)
        # print(x1.size())
        # print('s1size', s1.size())
        edge = self.eam(s4, s1)
        edge = F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=False)
        # # ---- Decoder 2 ----
        # x3_rfb = F.interpolate(x3_rfb, scale_factor=2,
        #                        mode='bilinear')  # NOTES: Sup-1 (bs, 1, 22, 22) -> (bs, 1, 44, 44)
        # conv4 = self.deconv_1(x3_rfb)
        # up_1 = torch.cat([conv4, x1], 1)  # 32+256 = 288#边界信息
        #
        # conv5 = self.deconv_2(up_1)
        # up_2 = conv5  # [bs, 64, 176, 176]
        # # up_2 = torch.cat([conv5, x1], 1)  # 64+? = #边界信息
        #
        # conv6 = self.deconv_3(up_2)
        #
        # lateral_map_2 = self.classifier(conv6)
        #
        # lateral_map_3 = self.out_conv(torch.cat((lateral_map_1, lateral_map_2), 1))

        return edge


def get_gt_bnd(gt):
    # get ground truth boundary using dilation
    gt = (gt > 0).astype(np.uint8).copy()
    bnd = np.zeros_like(gt).astype(np.uint8)
    for i in range(gt.shape[0]):
        _mask = gt[i]
        for j in range(1, _mask.max() + 1):
            _gt = (_mask == j).astype(np.uint8).copy()
            _gt_dil = dilation(_gt, disk(4))
            bnd[i][_gt_dil - _gt == 1] = 1
    return bnd


"""

Training 


"""


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def identify_axis(shape):
    """
    Helper function to enable loss function to be flexibly used for
    both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
    """
    # Three dimensional
    if len(shape) == 5:
        return [2, 3, 4]
    # Two dimensional
    elif len(shape) == 4:
        return [2, 3]
    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def to_onehot(y_pred, y_true):
    shp_x = y_pred.shape
    shp_y = y_true.shape
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            y_true = y_true.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(y_pred.shape, y_true.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y_true
        else:
            y_true = y_true.long()
            y_onehot = torch.zeros(shp_x, device=y_pred.device)
            y_onehot.scatter_(1, y_true, 1)
    return y_onehot


def get_tp_fp_fn_tn(net_output, gt, axes=None, square=False, weight=None):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    y_onehot = to_onehot(net_output, gt)

    if weight is None:
        weight = 1
    tp = net_output * y_onehot * weight
    fp = net_output * (1 - y_onehot) * weight
    fn = (1 - net_output) * y_onehot * weight
    tn = (1 - net_output) * (1 - y_onehot) * weight

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


def DiceLoss(y_pred, y_true):
    # first convert y_true to one-hot format
    smooth = 1e-8
    axis = identify_axis(y_pred.shape)
    y_pred = torch.sigmoid(y_pred)
    tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axis)
    intersection = 2 * tp + smooth
    union = 2 * tp + fp + fn + smooth
    dice = 1 - (intersection / union)
    return dice.mean()


losslist = []
losslist_edge = []


def train_d1(train_loader, encoder, decoder1, decoder2, en_optimizer, de1_optimizer, de2_optimizer, epoch):
    encoder.train()
    decoder1.train()
    decoder2.eval()

    save_path = 'Snapshots/{}/'.format(opt.train_save)
    loss_record, loss_record_edge = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):

        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        gts = gts.unsqueeze(1)
        # c = images.cpu().numpy()
        edges = torch.from_numpy(np.expand_dims(get_gt_bnd(gts.cpu().squeeze(1).numpy()), axis=1)).cuda()
        # a = gts.cpu().numpy()
        # ---- forward ----
        x1, x2, x3, x4 = encoder(images)
        edge = decoder2(x1, x4)
        lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_4 = decoder1(edge, x2, x3, x4)
        # ---- loss function ----
        loss4 = structure_loss(lateral_map_1, gts)
        loss3 = structure_loss(lateral_map_2, gts)
        loss2 = structure_loss(lateral_map_3, gts)
        loss1 = structure_loss(lateral_map_4, gts)
        # 0.2 * loss1 + 0.1 * loss2 + 0.2 * loss3 + 0.5 * loss4
        loss = loss4  # TODO: try different weights for loss
        lossedge = DiceLoss(edge, edges)
        # ---- encoder decoder1 backward ----
        en_optimizer.zero_grad()
        de1_optimizer.zero_grad()

        loss.backward()

        en_optimizer.step()
        de1_optimizer.step()
        # ---- recording loss ----

        loss_record.update(loss.data, opt.batchsize)
        loss_record_edge.update(lossedge.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Decoder1 Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  'lateral: {:0.4f}], edgeloss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(), loss_record_edge.show()))
    # 保存loss
    os.makedirs(save_path, exist_ok=True)
    losslist.append(loss_record.show().cpu().detach().numpy())
    np.savetxt(save_path + 'train_loss.csv', losslist, delimiter=',')
    losslist_edge.append(loss_record_edge.show().cpu().detach().numpy())
    np.savetxt(save_path + 'train_loss_edge.csv', losslist_edge, delimiter=',')

    if (epoch + 1) % 10 == 0:
        torch.save(encoder.state_dict(), save_path + 'encoder.pth')
        torch.save(decoder1.state_dict(), save_path + 'decoder1.pth')
        torch.save(decoder2.state_dict(), save_path + 'decoder2.pth')
        print('[Saving Snapshot:]', save_path + 'model_pth')


def train_d2(train_loader, encoder, decoder1, decoder2, en_optimizer, de1_optimizer, de2_optimizer, epoch):
    encoder.train()
    decoder1.train()
    decoder2.train()

    save_path = 'Snapshots/{}/'.format(opt.train_save)
    loss_record, loss_record_edge = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        gts = gts.unsqueeze(1)
        edges = torch.from_numpy(np.expand_dims(get_gt_bnd(gts.cpu().squeeze(1).numpy()), axis=1)).cuda()
        # ---- forward ----
        x1, x2, x3, x4 = encoder(images)
        edge = decoder2(x1, x4)
        lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_4 = decoder1(edge, x2, x3, x4)
        # ---- loss function ----
        loss4 = structure_loss(lateral_map_1, gts)
        loss3 = structure_loss(lateral_map_2, gts)
        loss2 = structure_loss(lateral_map_3, gts)
        loss1 = structure_loss(lateral_map_4, gts)
        # 0.2 * loss1 + 0.1 * loss2 + 0.2 * loss3 + 0.5 * loss4
        loss = loss4  # TODO: try different weights for loss
        lossedge = DiceLoss(edge, edges)
        # ---- encoder decoder2 backward ----
        en_optimizer.zero_grad()
        de2_optimizer.zero_grad()

        lossedge.backward()

        en_optimizer.step()
        de2_optimizer.step()
        loss_record.update(loss.data, opt.batchsize)
        loss_record_edge.update(lossedge.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Decoder2 Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  'lateral: {:0.4f}], edgeloss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(), loss_record_edge.show()))
    # # 保存loss
    # os.makedirs(save_path, exist_ok=True)
    # losslist.append(loss_record.show().cpu().detach().numpy())
    # np.savetxt(save_path + 'train_loss.csv', losslist, delimiter=',')
    # losslist_edge.append(loss_record_edge.show().cpu().detach().numpy())
    # np.savetxt(save_path + 'train_loss_edge.csv', losslist_edge, delimiter=',')
    #
    # if (epoch + 1) % 10 == 0:
    #     torch.save(encoder.state_dict(), save_path + 'encoder.pth')
    #     torch.save(decoder1.state_dict(), save_path + 'decoder1.pth')
    #     torch.save(decoder2.state_dict(), save_path + 'decoder2.pth')
    #     print('[Saving Snapshot:]', save_path + 'model_pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.05, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=25, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='train/Kvasir-Capsule', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='Kvasir-Capsule/nextmodel_de2_v3')
    parser.add_argument('--OptPeriod', type=int,
                        default=1, help='decoder2 train period')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    encoder = Encoder().cuda()
    decoder1 = Decoder1().cuda()
    decoder2 = Decoder2().cuda()

    # ---- flops and params ----
    en_params = encoder.parameters()
    de1_params = decoder1.parameters()
    de2_params = decoder2.parameters()
    en_optimizer = torch.optim.Adam(en_params, opt.lr)
    de1_optimizer = torch.optim.Adam(de1_params, opt.lr)
    de2_optimizer = torch.optim.Adam(de2_params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(en_optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        adjust_lr(de1_optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        adjust_lr(de2_optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train_d1(train_loader, encoder, decoder1, decoder2, en_optimizer, de1_optimizer, de2_optimizer, epoch)
        if epoch % opt.OptPeriod == 0 and opt.epoch-epoch >= 10:
            train_d2(train_loader, encoder, decoder1, decoder2, en_optimizer, de1_optimizer, de2_optimizer, epoch)
