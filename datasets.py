import os, glob
import random
from PIL import Image

import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DataSeq(Dataset):
    def __init__(self, input_dir='inputs'):
        """Dataset providing CIFAR10 grayscale images as inputs"""
        self.files = sorted(glob.glob(os.path.abspath(os.path.join(input_dir, '**', '*.jpg')), recursive=True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_data = self.files[idx]
        return image_data


class AugmentedData(Dataset):
    def __init__(self, input_dir='inputs', input_image_size=(100,100), min_offset=0, max_offset=8, min_spacing=2, max_spacing=6, min_pixels=144):
        self.dataset = DataSeq(input_dir)
        self.transform_chain = transforms.Compose([transforms.Resize(size=input_image_size),
                                                    # transforms.RandomCrop(size=input_image_size),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomVerticalFlip(p=0.5),
                                                    transforms.ColorJitter(brightness=0.5, hue=0.3)
                                                    ])
        self.min_offset, self.max_offset = min_offset, max_offset
        self.min_spacing, self.max_spacing = min_spacing, max_spacing
        self.min_pixels = min_pixels


    def normalize(self, image_array):
        # Pixel-wise Normalization
        mean = image_array.mean()
        std = image_array.std()
        image_array[:] -= mean
        image_array[:] /= std
        return image_array

    def random_off_space(self):
        return ((random.randint(0,8), random.randint(0,8)), (random.randint(2,6), random.randint(2,6)))
        # return ((random.randint(self.min_offset,self.max_offset), random.randint(self.min_offset,self.max_offset)),
        #         (random.randint(self.min_spacing, self.max_spacing), random.randint(self.min_spacing, self.max_spacing)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        offset, spacing = self.random_off_space()
        image_array = self.transform_chain(Image.open(self.dataset[idx]))

        image_array = np.asarray(image_array, dtype=np.float32)
        # image_array = self.normalize(image_array)
        image_array = np.transpose(image_array, (2, 0, 1))
        input_array = np.copy(image_array)

        known_array = np.zeros_like(image_array)
        known_array[:, offset[1]::spacing[1], offset[0]::spacing[0]] = 1

        known_pixels = np.sum(known_array[0], dtype=np.uint32)
        if known_pixels < 144:
            raise ValueError(f"The number of known pixels after removing must be at "
                             f"least {144} but is {known_pixels}")

        target_array = image_array[known_array == 0].copy()
        input_array[known_array == 0] = 0

        return input_array, known_array, target_array, image_array




def collater(batch):
    inputs = torch.zeros((len(batch), 3, 100, 100), dtype=torch.float32)
    outputs = torch.zeros((len(batch), 3, 100, 100), dtype=torch.float32)
    knowns = torch.zeros((len(batch), 3, 100, 100), dtype=torch.float32)

    targets = [torch.tensor(i[2], dtype=torch.float32) for i in batch]

    for idx, (input_array, known_array, target_array, image_array) in enumerate(batch):
        inputs[idx] = torch.from_numpy(input_array)
        outputs[idx] = torch.from_numpy(image_array)
        knowns[idx] = torch.from_numpy(known_array)
    return inputs, targets, outputs, knowns
