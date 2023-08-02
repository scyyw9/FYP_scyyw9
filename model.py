from collections import OrderedDict
from swinTransformer.swin_model import window_partition

import torch
import torch.nn as nn
import torchvision
import numpy as np


class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]

    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat


class SwinTransformerFPN(nn.Module):
    def __init__(self, model, window_size):
        super(SwinTransformerFPN, self).__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2

        children = list(model.children())
        # patch partition and linear embedding
        self.patch_embed = children[0]
        self.pos_drop = children[1]
        # layer 0 (stage 0)
        self.layer0 = children[2][0]

        # swinTransformer blocks in stage 1
        self.layer1_blocks = children[2][1].blocks
        self.layer1_downsample = children[2][1].downsample

        # swinTransformer blocks in stage 2
        self.layer2_blocks = children[2][2].blocks
        self.layer2_downsample = children[2][2].downsample

        # swinTransformer blocks in stage 3
        self.layer3_blocks = children[2][3].blocks

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        feat = OrderedDict()

        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        # stage 0 (layer0 trans + stage 1 downsample)
        x, H, W = self.layer0(x, H, W)  # H/4, W/4, C

        # stage 1 trans
        attn_mask = self.create_mask(x, H, W)
        for blk in self.layer1_blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)  # H/8, W/8, 2C

        feat_map3 = x.view(x.shape[0], H, W, x.shape[2])
        feat['map3'] = feat_map3

        # stage 2 downsample
        x = self.layer1_downsample(x, H, W)
        H, W = (H + 1) // 2, (W + 1) // 2
        # stage 2 trans
        attn_mask = self.create_mask(x, H, W)
        for blk in self.layer2_blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)  # H/16, W/16, 4C

        feat_map4 = x.view(x.shape[0], H, W, x.shape[2])
        feat['map4'] = feat_map4

        return feat


class CountRegressor(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, (7, 7), padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, (5, 5), padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 1, (1, 1)),
            nn.ReLU(),
        )

    def forward(self, im):
        num_sample = im.shape[0]
        if num_sample == 1:
            output = self.regressor(im.squeeze(0))
            if self.pool == 'mean':
                output = torch.mean(output, dim=0, keepdim=True)
                return output
            elif self.pool == 'max':
                output, _ = torch.max(output, 0, keepdim=True)
                return output
        else:
            for i in range(0, num_sample):
                output = self.regressor(im[i])
                if self.pool == 'mean':
                    output = torch.mean(output, dim=0, keepdim=True)
                elif self.pool == 'max':
                    output, _ = torch.max(output, 0, keepdim=True)
                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output, output), dim=0)
            return Output


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
