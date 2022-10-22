import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import compare_ssim as sk_ckpt_ssim

import os
import glob
import random

import torch
import glob
import random

import torch

if torch.cuda.is_available():
    torch.cuda.current_device()

import torchvision.transform.functional as TF
from torch.utils.data import Dataset, Dataloader, Subset
from torchvision import transform, utils

import json


class PairdeDataAugmentation:
    def __init__(
        self,
        img_size,
        with_random_hflip=False,
        with_random_vflip=False,
        with_random_rot90=False,
        with_random_rot180=False,
        with_random_rot270=False,
        with_random_crop=False,
        with_random_brightness=False,
        with_random_gamma=False,
        with_random_saturation=False,
    ):
        self.img_size = img_size
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot90 = with_random_rot90
        self.with_random_rot180 = with_random_rot180
        self.with_random_rot270 = with_random_rot270
        self.with_random_crop = with_random_crop
        self.with_random_brightness = with_random_brightness
        self.with_random_gamma = with_random_gamma
        self.with_random_saturation = with_random_saturation


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_config(path_to_json=r"./config.json"):
    with open(path_to_json) as f:
        data = json.load(f)
    args = Struct(**data)
    return args


def clip_01(x):
    x[x > 1.0] = 1.0
    x[x < 0] = 0
    return x


def cpt_pxl_cls_acc(pred_idx, target):
    pred_idx = torch.reshape(pred_idx, [-1])
    target = torch.reshape(target, [-1])
    return torch.mean((pred_idx.int() == target.int()).float())


def cpt_batch_psnr(img, img_gt, PIXEL_MAX):
    mse = torch.mean((img - img_gt) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return torch.mean(psnr)


def cpt_psnr(img, img_gt, PIXEL_MAX):
    mse = np.mean((img - img_gt) ** 2)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


def cpt_rgb_ssim(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    SSIM = 0
    for i in range(3):
        tmp = img[:, :, i]
        tmp_gt = img_gt[:, :, i]
        ssim = sk_ckpt_ssim(tmp, tmp_gt)
        SSIM = SSIM + ssim
    return SSIM / 3.0


def cpt_ssim(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    return sk_ckpt_ssim(img, img_gt)
