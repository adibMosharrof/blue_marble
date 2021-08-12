data_path = "/localdisk0/SCRATCH/amo304/blue_marble"

import os
import numpy as np
import uuid

# from IPython.display import Image
import PIL
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = None
import random
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


class BlueMarble:
    crop_shape = [28, 28]
    data_path = "data"
    imgs = {}
    results_dir = "results"

    def __init__(self):
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        if socket.gethostname() == "loon":
            self.data_path = "/localdisk0/SCRATCH/amo304/blue_marble/data"

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
        # plt.show()
        # plt.savefig(os.path.join(path, str(uuid.uuid4())))
        plt.close(f)

    def plot_loss_graph(self, x, y, xlabel="epoch number", ylabel="loss value"):
        plt.plot(x, y)
        plt.xlabel = xlabel
        plt.ylabel = ylabel
        plt.title("Loss history over iterations")
        plt.savefig(os.path.join(self.results_dir, "Loss_history"))

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
        self.load_images()
        data_loader = DataLoader(
            BlueMarbleDataLoader(
                os.listdir(self.data_path),
                self.imgs,
                self.crop_shape,
                total_data_size=100,
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
        model = self.train(net, data_loader, optimizer, criterion, epochs=5)
        print("finished training")
        torch.save(model.state_dict(), os.path.join(self.results_dir, "model.pt"))
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
        self.load_images()
        test_data_loader = self.get_test_data_loader()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(self.results_dir, timestr)
        # os.mkdir(out_dir)
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
            self.display([a, b], path=timestr, text=text)

        # self.plot_histogram(similar, different, path=out_dir)

    def get_patches(self, img, crop_shape=None):
        size = crop_shape or self.crop_shape[0]
        img_t = torch.tensor(np.array(img, "float"))
        patches = (
            img_t.unfold(0, size, size).unfold(1, size, size).unfold(2, size, size)
        )
        return patches

    def visualize_patch(self):
        model = self.load_model()
        test_data_loader = self.get_test_data_loader(
            crop_shape=[508, 508], total_data_size=2
        )

        for i, data in enumerate(test_data_loader):
            img_large, img2, label = data
            patches = self.get_patches(img_large)
            out_large, _ = model(img_large, img2)
            out_small_list = []
            for patch in patches:
                out_small, _ = model(patch, torch.rand(patch.shape))
                out_small_list.append(out_small)
            a = 1

    def plot_one(self, data, text):
        plt.figure(figsize=(8, 6))
        plt.hist(data, alpha=0.5, label=text)
        plt.ylabel("Frequency", size=14)
        plt.xlabel("Euclidian Distance", size=14)
        plt.legend(loc="upper right")
        plt.title(text)
        plt.savefig(os.path.join(self.results_dir, text))

    def plot_histogram(self, similar, different, path=""):
        plt.figure(figsize=(8, 6))
        plt.hist(similar, alpha=0.5, label="similar")
        plt.hist(different, alpha=0.5, label="different")
        plt.ylabel("Frequency", size=14)
        plt.xlabel("Euclidian Distance", size=14)
        plt.legend(loc="upper right")
        plt.title("Euclidian score distribution")
        plt.savefig(os.path.join(path, "eucledian score histogram"))

    # visualize_patch()


bm = BlueMarble()
bm.run()
# bm.test()
