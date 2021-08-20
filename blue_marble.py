data_path = "/localdisk0/SCRATCH/amo304/blue_marble"

import os
import numpy as np
import uuid

# from IPython.display import Image
import PIL
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = None
import random
from math import sqrt
from random import randint
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms.functional import crop
from siamese_network import ContrastiveLoss, SiameseNetwork
from blue_marble_data_loader import BlueMarbleDataLoader
from torch.utils.data import DataLoader
import time
import pickle
import socket
from pathlib import Path
import heatmap
import seaborn as sns
import logging


class BlueMarble:
    crop_shape = [28, 28]
    data_path = "data"
    out_dir = ""
    imgs = {}
    results_dir = "results"
    logger = None

    def __init__(self):
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        if socket.gethostname() == "loon":
            self.data_path = "/localdisk0/SCRATCH/amo304/blue_marble/data"
        self.load_images()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.out_dir = os.path.join(self.results_dir, timestr)
        os.mkdir(self.out_dir)
        log_file = os.path.join(self.out_dir, "log.txt")
        self.logger = logging.getLogger("default")
        self.logger.basicConfig(filename=log_file, encoding="utf-8")
        self.logger.info(f"crop shape {self.crop_shape}")

    def load_images(self):
        file_name = "imgs.pickle"
        project_root = Path(self.data_path).parent.absolute()
        pickle_file_path = os.path.join(project_root, file_name)
        if os.path.isfile(pickle_file_path):
            with open(pickle_file_path, "rb") as file:
                print("starting IO")
                self.imgs = pickle.load(file)
                print("ending IO")
            return

        for name in os.listdir(self.data_path):
            self.imgs[name] = Image.open(os.path.join(self.data_path, name)).convert(
                "RGB"
            )

        with open(pickle_file_path, "wb") as file:
            file = open(pickle_file_path, "wb")
            pickle.dump(self.imgs, file)

    def display(self, imgs, text=None, path=""):
        n = len(imgs)
        f = plt.figure()
        for i, img in enumerate(imgs):
            f.add_subplot(1, n, i + 1)
            plt.imshow(img)
        if text:
            f.suptitle(text, bbox={"facecolor": "red", "alpha": 0.5})
        plt.show()
        # plt.savefig(os.path.join(path, str(uuid.uuid4())))
        plt.close(f)

    def train(self, net, data_loader, optimizer, criterion, epochs=2):
        counter = []
        loss_history = []
        iteration_number = 0

        for epoch in range(0, epochs):
            for i, data in enumerate(data_loader, 0):
                img0, img1, label = data.values()
                # img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                optimizer.zero_grad()
                output1, output2 = net(img0, img1)
                loss_contrastive = criterion(output1, output2, label)
                loss_contrastive.backward()
                optimizer.step()
                if i % 50 == 0:
                    print(
                        "Epoch number {}\n Current loss {}\n".format(
                            epoch, loss_contrastive.item()
                        )
                    )
                    iteration_number += 10
            counter.append(epoch)
            loss_history.append(loss_contrastive.item())
        self.plot_loss_graph(counter, loss_history)
        return net

    def run(self):
        data_loader = DataLoader(
            BlueMarbleDataLoader(
                os.listdir(self.data_path),
                self.imgs,
                self.crop_shape,
                total_data_size=1000,
            ),
            num_workers=8,
            batch_size=25,
        )

        net = SiameseNetwork()
        criterion = ContrastiveLoss()
        optimizer = torch.optim.RMSprop(
            net.parameters(),
            lr=1e-4,
            alpha=0.99,
            eps=1e-8,
            weight_decay=0.0005,
            momentum=0.9,
        )
        print("starting training")
        model = self.train(net, data_loader, optimizer, criterion, epochs=100)
        print("finished training")
        torch.save(model.state_dict(), os.path.join(self.results_dir, "model.pt"))
        torch.save(model.state_dict(), os.path.join(self.out_dir, "model.pt"))
        print("model saved")

    def load_model(self):
        model = SiameseNetwork()
        model.load_state_dict(torch.load(os.path.join(self.results_dir, "model.pt")))
        return model

    def get_test_data_loader(self, crop_shape=None, total_data_size=60):
        if crop_shape is None:
            crop_shape = self.crop_shape

        test_data_loader = DataLoader(
            BlueMarbleDataLoader(
                os.listdir(self.data_path),
                self.imgs,
                crop_shape,
                total_data_size=total_data_size,
            ),
            num_workers=1,
            batch_size=1,
        )

        return test_data_loader

    def test(self):
        model = self.load_model()
        test_data_loader = self.get_test_data_loader(total_data_size=100)
        similar = []
        different = []
        for i, data in enumerate(test_data_loader):
            img0, img1, label = data.values()
            output1, output2 = model(img0, img1)
            eucledian_distance = round(F.pairwise_distance(output1, output2).item(), 2)
            label_text = "similar"
            if label:
                different.append(eucledian_distance)
                label_text = "different"
            else:
                similar.append(eucledian_distance)

            text = f"Dissimilarity: {round(eucledian_distance,2)} Label: {label_text}"
            a = torch.squeeze(img0, 0).permute(1, 2, 0)
            b = torch.squeeze(img1, 0).permute(1, 2, 0)
            # self.display([a, b], path=timestr, text=text)

        self.plot_histogram(similar, different)

    def get_patches(self, img, crop_shape=None):
        size = crop_shape or self.crop_shape[0]
        img_t = torch.tensor(np.array(img, "float"))
        patches = img_t.unfold(0, 3, 3).unfold(1, size, size).unfold(2, size, size)
        print(patches.shape)
        return patches

    def visualize_patch(self):
        model = self.load_model()

        # large_crop_shape = 1008
        large_crop_shape = 500
        small_crop_shape = 10
        test_data_loader = self.get_test_data_loader(
            crop_shape=[large_crop_shape, large_crop_shape], total_data_size=6
        )
        for _, data in enumerate(test_data_loader):
            cropper = T.CenterCrop(size=small_crop_shape)
            imgs_large, imgs2, labels = data.values()
            out_large, _ = model(imgs_large, imgs2)
            imgs_small = cropper(imgs_large)
            out_small, _ = model(imgs_small, imgs_small)
            out_large_size = out_large.shape[1]
            out_small_size = out_small.shape[1]
            distances = []
            for i in range(0, out_large_size, out_small_size):
                patch = out_large[:, i : i + out_small_size]
                distances.append(F.pairwise_distance(out_small, patch))

            dims = int(sqrt(len(distances)))
            scores = torch.tensor(distances).reshape([dims, dims])
            img_large = imgs_large.squeeze(0).permute(1, 2, 0).byte()
            img_small = imgs_small.squeeze(0).permute(1, 2, 0).byte()
            self.plot_heat_map(scores, img_small, img_large)

    def plot_heat_map(self, data, center, orig):

        fig = plt.figure(figsize=(25, 25))
        ax = fig.add_subplot(211)
        axis_labels = np.arange(len(data))
        im, cbar = heatmap.heatmap(data, axis_labels, axis_labels)
        # texts = heatmap.annotate_heatmap(im)
        fig.tight_layout()
        plt.rc("font", size=10)
        plot_name = f"heat_map_{str(uuid.uuid4())}"
        plt.title("Euclidian distances from center crop", fontsize=10)
        ax = fig.add_subplot(223)
        plt.title("Center crop", fontsize=10)
        ax.imshow(center)
        ax = fig.add_subplot(224)
        plt.title("Original image", fontsize=10)
        ax.imshow(orig)
        plt.savefig(os.path.join(self.out_dir, plot_name), dpi=600)
        fig.clf()
        plt.clf()
        plt.cla()
        plt.close()

    def plot_histogram(self, similar, different):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
        ax.hist(similar, alpha=0.5, label="similar")
        ax.hist(different, alpha=0.5, label="different")
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Euclidian Distance")
        # plt.ylabel("Frequency", size=14)
        # plt.xlabel("Euclidian Distance", size=14)
        plt.legend(loc="upper right")
        # plt.title("Euclidian score distribution")
        plt.savefig(os.path.join(self.out_dir, "eucledian score histogram"))
        fig.clf()
        plt.clf()
        plt.cla()
        plt.close()

    def plot_loss_graph(self, x, y, xlabel="epoch number", ylabel="loss value"):
        fig, ax = plt.subplots(1)
        plt.plot(x, y)
        plt.xlabel = xlabel
        plt.ylabel = ylabel
        plt.title("Loss history over epochs")
        plt.savefig(os.path.join(self.out_dir, "Loss_history"))
        fig.clf()
        plt.clf()
        plt.cla()
        plt.close(fig)


bm = BlueMarble()
bm.run()
bm.test()
# bm.visualize_patch()
