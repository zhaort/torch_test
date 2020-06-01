from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image


transform = transforms.Compose([
    transforms.ToTensor()
])

content = load_image('C:\\Users\\45569\\Pictures\\Saved Pictures\\2.jpg',
                     transform, max_size=224)
style = load_image('C:\\Users\\45569\\Pictures\\Saved Pictures\\2.jpeg',
                   transform, shape=[content.size(3), content.size(2)])

print(content.shape)
print(style.shape)


def show_image(image):
    transforms.ToPILImage()(image.squeeze(0)).show()


show_image(content)
