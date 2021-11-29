import numpy as np
import random
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
from PIL import Image


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpts=None, mask=None):
        for t in self.transforms:
            img, kpts, mask = t(img, kpts, mask)
        return img, kpts, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):

    def __call__(self, img, kpts, mask):
        return np.asarray(img).astype(np.float32) / 255., kpts, mask


class NormalizeTraining(object):
    # this is the one with polarized images

    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img, kpts, mask):
        # added calculation of the mean values of the 2 additional channels on the fly
        # usually values are mean=[0.002248, 0.002258] and std=[0.001851, 0.001850]
        n_mean = self.mean + [np.mean(img[:, :, 3]), np.mean(img[:, :, 4])]
        n_std = self.std + [np.std(img[:, :, 3]), np.std(img[:, :, 4])]
        img -= n_mean
        img /= n_std
        if self.to_bgr:  # we put the channels first, then the rows and columns
            img = img.transpose(2, 0, 1).astype(np.float32)
        return img, kpts, mask


class NormalizeTest(object):
    # this is the one for testing with only RGB

    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img, kpts, mask):
        # added calculation of the mean values of the 2 additional channels on the fly
        # usually values are mean=[0.002248, 0.002258] and std=[0.001851, 0.001850]
        img -= self.mean
        img /= self.std
        if self.to_bgr:  # we put the channels first, then the rows and columns
            img = img.transpose(2, 0, 1).astype(np.float32)
        return img, kpts, mask


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def __call__(self, image, kpts, mask):
        image[:, :, :3] = np.asarray(self.color_jitter(Image.fromarray(np.ascontiguousarray(image[:, :, :3], np.uint8))))
        return image, kpts, mask


class RandomBlur(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, kpts, mask):
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])
            image = cv2.GaussianBlur(image, (sigma, sigma), 0)
        return image, kpts, mask


def make_transforms(cfg, is_train):
    if is_train is True:
        transform = Compose(
            [
                RandomBlur(0.5),
                ColorJitter(0.1, 0.1, 0.05, 0.05),
                ToTensor(),
                NormalizeTraining(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_bgr=True),
            ]
        )
    else:
        if cfg.pol_inference:
            transform = Compose(
                [
                    ToTensor(),
                    # change to NormalizeTest for RGB only inference
                    NormalizeTraining(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_bgr=True),
                ])
        else:
            transform = Compose(
                [
                    ToTensor(),
                    # change to NormalizeTest for RGB only inference
                    NormalizeTest(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_bgr=True),
                ])

    return transform
