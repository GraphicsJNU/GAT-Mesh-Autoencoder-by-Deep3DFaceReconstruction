import torch
import torch.nn as nn
from torch.nn import Module, init, Sequential, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, Dropout, ReLU, Tanh
from torchvision import transforms
import torchvision.transforms.functional as F


class UvGan(Module):
    def __init__(self):
        super(UvGan, self).__init__()
        self.size = 256, 256
        self.resize = transforms.Resize(256, F.InterpolationMode.NEAREST)
        self.flip = transforms.RandomHorizontalFlip()

        def generateDownSampleLayer(in_channels, out_channels, kernel_size, apply_batchnorm=True):
            result = Sequential()
            conv_layer = Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=2,
                                padding='same', bias=False)
            init.normal_(conv_layer.weight, mean=0., std=0.02)
            result.append(conv_layer)

            if apply_batchnorm:
                result.add(BatchNorm2d(out_channels))

            result.add(LeakyReLU())

            return result

        def generateUpSampleLayer(in_channels, out_channels, kernel_size, apply_dropout=False):
            result = Sequential()
            conv_layer = ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=2,
                                         padding=1, bias=False)
            init.normal_(conv_layer.weight, mean=0., std=0.02)
            result.add(conv_layer)

            result.add(BatchNorm2d(out_channels))

            if apply_dropout:
                result.add(Dropout(0.5))

            result.add(ReLU())

            return result

        self.down_stack = [
            generateDownSampleLayer(6, 64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            generateDownSampleLayer(64, 128, 4),  # (bs, 64, 64, 128)
            generateDownSampleLayer(128, 256, 4),  # (bs, 32, 32, 256)
            generateDownSampleLayer(256, 512, 4),  # (bs, 16, 16, 512)
            generateDownSampleLayer(512, 512, 4),  # (bs, 8, 8, 512)
            generateDownSampleLayer(512, 512, 4),  # (bs, 4, 4, 512)
            generateDownSampleLayer(512, 512, 4),  # (bs, 2, 2, 512)
            generateDownSampleLayer(512, 512, 4),  # (bs, 1, 1, 512)
        ]

        self.up_stack = [
            generateUpSampleLayer(512, 512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            generateUpSampleLayer(512, 512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            generateUpSampleLayer(512, 512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            generateUpSampleLayer(512, 512, 4),  # (bs, 16, 16, 1024)
            generateUpSampleLayer(512, 256, 4),  # (bs, 32, 32, 512)
            generateUpSampleLayer(526, 128, 4),  # (bs, 64, 64, 256)
            generateUpSampleLayer(128, 64, 4),  # (bs, 128, 128, 128)
        ]

        OUTPUT_CHANNELS = 3
        conv_layer = ConvTranspose2d(in_channels=6, out_channels=OUTPUT_CHANNELS,
                                     kernel_size=4, stride=2,
                                     padding=1, bias=False)
        init.normal_(conv_layer.weight, mean=0., std=0.02)
        self.last = Sequential(
            conv_layer,
            Tanh()
        )

    def forward(self, img):
        def normalize(image):
            return (image / 127.5) - 1

        def randomJitter(image):
            image = self.resize(image)
            flip_image = self.flip(image)
            image = torch.concat([image, flip_image], dim=1)

            return image

        img = randomJitter(img)
        img = normalize(img)

        # Downsampling through the model
        x = img
        skips = []
        for down in self.down_stack:
            img = down(img)
            skips.append(img)
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            img = up(img)
            img = torch.concat([img, skip], dim=1)
        img = self.last(img)

        return img
