import torch
from torch.utils.data import Dataset
import random
import PIL
import os
from PIL import Image
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import uuid


class BlueMarbleDataLoader(Dataset):
    data_path = "data"

    def __init__(self, img_names, imgs, crop_shape, total_data_size=50) -> None:
        super().__init__()
        self.img_names = img_names
        self.crop_shape = crop_shape
        self.total_data_size = total_data_size
        self.imgs = imgs

    def __getitem__(self, index):
        img1_name = random.choice(self.img_names)
        img2_name = self._get_img2_name(img1_name)

        should_get_negative_example = random.randint(0, 1)

        img1 = self._read_image(img1_name)
        img2 = self._read_image(img2_name)
        box = self._get_crop_box(img1.size)
        crop1 = self._my_crop(img1, box)

        if should_get_negative_example:
            box = self._get_crop_box(img1.size)
        crop2 = self._my_crop(img2, box)

        return {
            "image_1": torch.from_numpy(np.expand_dims(crop1, 0)).float(),
            "image_2": torch.from_numpy(np.expand_dims(crop2, 0)).float(),
            "label": should_get_negative_example,
        }

    def __len__(self):
        return self.total_data_size

    def _get_img2_name(self, used_name):
        while True:
            name = random.choice(self.img_names)
            if name is not used_name:
                return name

    def _read_image(self, img_name):
        return self.imgs[img_name]
        # return Image.open(os.path.join(self.data_path, img_name)).convert("L")

    def _my_crop(self, img, box):
        # top, left, height, width = box
        c_img = img.crop(box)
        return np.asarray(c_img)

    def _get_crop_box(self, img_shape):
        rand_x = randint(0, img_shape[0] - self.crop_shape[0])
        rand_y = randint(0, img_shape[1] - self.crop_shape[1])

        left = rand_x
        top = rand_y
        right = rand_x + self.crop_shape[0]
        bottom = rand_y + self.crop_shape[1]

        box = (left, top, right, bottom)
        return box

    def display(self, imgs):
        n = len(imgs)
        f = plt.figure()
        for i, img in enumerate(imgs):
            f.add_subplot(1, n, i + 1)
            plt.imshow(img)
        plt.show()
        plt.savefig(str(uuid.uuid4()))
        plt.close(f)
