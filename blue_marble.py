data_path = "/localdisk0/SCRATCH/amo304/blue_marble"

import os
import numpy as np
import uuid

# from IPython.display import Image
import PIL
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = None
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

crop_shape = [28, 28]
data_path = "data"
results_dir = "results"
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

imgs = {}
for name in os.listdir(data_path):
    imgs[name] = Image.open(os.path.join(data_path, name)).convert("L")


def display(imgs, text=None, path=""):
    n = len(imgs)
    f = plt.figure()
    for i, img in enumerate(imgs):
        f.add_subplot(1, n, i + 1)
        plt.imshow(img)
    if text:
        f.suptitle(text, bbox={"facecolor": "red", "alpha": 0.5})
    # plt.show()
    plt.savefig(os.path.join(path, str(uuid.uuid4())))
    plt.close(f)


def plot_graph(x, y, xlabel="num iterations", ylabel="loss value"):
    plt.plot(x, y)
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.title("Loss history over iterations")
    plt.savefig(os.path.join(results_dir, "Loss_history"))


def train(net, data_loader, optimizer, criterion, epochs=2):
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
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    plot_graph(counter, loss_history)
    return net


def run():
    data_loader = DataLoader(
        BlueMarbleDataLoader(
            os.listdir(data_path), imgs, crop_shape, total_data_size=10000
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
    model = train(net, data_loader, optimizer, criterion, epochs=50)
    print("finished training")
    torch.save(model.state_dict(), os.path.join(results_dir, "model.pt"))
    print("model saved")


def test():
    model = SiameseNetwork()
    model.load_state_dict(torch.load(os.path.join(results_dir, "model.pt")))

    test_data_loader = DataLoader(
        BlueMarbleDataLoader(
            os.listdir(data_path), imgs, crop_shape, total_data_size=60
        ),
        num_workers=1,
        batch_size=1,
    )
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(os.path.join(results_dir, timestr))
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
        display([a, b], path=timestr, text=text)

    plot_histogram(similar, different, path=timestr)


def plot_one(data, text):
    plt.figure(figsize=(8, 6))
    plt.hist(data, alpha=0.5, label=text)
    plt.ylabel("Frequency", size=14)
    plt.xlabel("Euclidian Distance", size=14)
    plt.legend(loc="upper right")
    plt.title(text)
    plt.savefig(os.path.join(results_dir, text))


def plot_histogram(similar, different, path=""):
    plt.figure(figsize=(8, 6))
    plt.hist(similar, alpha=0.5, label="similar")
    plt.hist(different, alpha=0.5, label="different")
    plt.ylabel("Frequency", size=14)
    plt.xlabel("Euclidian Distance", size=14)
    plt.legend(loc="upper right")
    plt.title("Euclidian score distribution")
    plt.savefig(os.path.join(path, "eucledian score histogram"))


test()
