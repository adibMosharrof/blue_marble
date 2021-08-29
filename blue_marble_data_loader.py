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

random.seed(3)


class BlueMarbleDataLoader(Dataset):
    data_path = "data"

    def __init__(
        self, img_names, imgs, crop_shape, total_data_size=50, loc=None
    ) -> None:
        super().__init__()
        self.img_names = img_names
        self.crop_shape = crop_shape
        self.total_data_size = total_data_size
        self.imgs = imgs
        self.loc = loc

    def __getitem__(self, index):
        img1_name = random.choice(self.img_names)
        img2_name = self._get_img2_name(img1_name)

        should_get_negative_example = random.randint(0, 1)
        boxes = []
        img1 = self._read_image(img1_name)
        img2 = self._read_image(img2_name)
        box = self._get_crop_box(img1.size)
        # boxes.append(box)
        crop1 = self._my_crop(img1, box)

        if should_get_negative_example:
            box = self._get_crop_box(img1.size)
            boxes.append(box)
        crop2 = self._my_crop(img2, box)

        return {
            "image_1": torch.from_numpy(crop1).permute(2, 0, 1).float(),
            "image_2": torch.from_numpy(crop2).permute(2, 0, 1).float(),
            "label": should_get_negative_example,
            # "box_1": boxes[0],
            # "box_2": boxes[1],
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
        return np.asarray(c_img, "float")

    def _get_crop_box(self, img_shape):
        rand_x, rand_y = self._get_loc(img_shape)

        left = rand_x
        top = rand_y
        right = rand_x + self.crop_shape[0]
        bottom = rand_y + self.crop_shape[1]

        box = (left, top, right, bottom)
        return box

    def _get_loc(self, img_shape):
        if not self.loc:
            return (
                randint(0, img_shape[0] - self.crop_shape[0]),
                randint(0, img_shape[1] - self.crop_shape[1]),
            )
        loc = random.choice(self.loc)
        return loc[0], loc[1]

    def display(self, imgs):
        n = len(imgs)
        f = plt.figure()
        for i, img in enumerate(imgs):
            f.add_subplot(1, n, i + 1)
            plt.imshow(img)
        plt.show()
        plt.savefig(str(uuid.uuid4()))
        plt.close(f)
