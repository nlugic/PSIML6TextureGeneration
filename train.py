import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from discriminator import SGANDiscrimantor
from generator import SGANGenerator
from helpers import get_train_dataset, init_sgan_weights, save_tensor_as_image

layers = 5
random_size = 9
random_channels = 50
batch_size = 32
total_epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

gen = SGANGenerator(layers, random_channels).to(device)
gen.apply(init_sgan_weights)
dis = SGANDiscrimantor(layers).to(device)
dis.apply(init_sgan_weights)

dataset_folder = "./train_textures/"
dataset_iter = get_train_dataset(dataset_folder, device, size = (random_size - 1) * 2 ** layers + 1, batch_size = batch_size)

writer = SummaryWriter()
z9 = torch.rand(batch_size, random_channels, 9, 9).to(device) * 2.0 - 1.0
z20 = torch.rand(batch_size, random_channels, 20, 20).to(device) * 2.0 - 1.0

curr_epoch = 0

while curr_epoch < total_epochs:
    curr_epoch += 1
    print("Epoch", curr_epoch)

    n_iter = 50 # oni imaju 100 iteracija, ali u jednoj treniraju ili gen ili dis, a mi oba
    for i, img_real in enumerate(tqdm(dataset_iter, total = n_iter)):
        if i >= n_iter:
            break

        # Generate image
        z = torch.rand(batch_size, random_channels, random_size, random_size).to(device) * 2.0 - 1.0
        img_fake = gen(z)

        # Train discriminator
        dis.training_step(img_real, img_fake)

        # Train generator
        gen.training_step(dis(img_fake))

    writer.add_scalar("Loss/Generator", np.mean(gen.loss_history), curr_epoch)
    writer.add_scalar("Loss/Discriminator", np.mean(dis.loss_history), curr_epoch)
    writer.add_images("Generated images/l9", gen(z9).detach()[:4] / 2 + 0.5, curr_epoch)
    writer.add_images("Generated images/l20", gen(z20).detach()[:4] / 2 + 0.5, curr_epoch)
