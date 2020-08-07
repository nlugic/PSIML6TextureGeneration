from tqdm import tqdm
import torch
import numpy as np

from discriminator import SGANDiscrimantor
from generator import SGANGenerator
from helpers import get_train_dataset, init_sgan_weights

layers = 5
nz = 50
l = 9
batch_size = 32

device = torch.device("cpu")

gen = SGANGenerator(nz, layers).to(device)
gen.layers.apply(init_sgan_weights)
dis = SGANDiscrimantor(layers).to(device)
dis.layers.apply(init_sgan_weights)

dataset_folder = "train_textures/"
dataset_iter = get_train_dataset(dataset_folder, device, size=l * 2**layers)

loss_funct_g = lambda pred: -torch.mean(torch.log(pred))
loss_funct_d = lambda pred_real, pred_fake: -torch.mean(torch.log(1 - pred_fake)) - torch.mean(torch.log(pred_real))

optim_g = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5)
optim_d = torch.optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5)


epoch = 0
while True:
    epoch += 1
    print("Epoch", epoch)

    loss_list_d = []
    loss_list_g = []

    n_iter = 1 # oni imaju 100 iteracija, ali u jednoj treniraju ili gen ili dis, a mi oba
    for i, img_real in enumerate(tqdm(dataset_iter, total=n_iter)):
        if i >= n_iter:
            break

        z = torch.rand(batch_size, nz, l, l) * 2 - 1
        
        # Train generator
        gen.zero_grad()
        img_fake = gen(z)
        pred_fake = dis(img_fake)

        loss_g = loss_funct_g(pred_fake)
        loss_list_g.append(loss_g.item())
        loss_g.backward()
        optim_g.step()

        # Train discriminator
        dis.zero_grad()
        img_fake = gen(z)
        pred_fake = dis(img_fake)
        pred_real = dis(img_real)

        loss_d = loss_funct_d(pred_real, pred_fake)
        loss_list_d.append(loss_d.item())
        loss_d.backward()
        optim_d.step()

    print("Loss G:", np.mean(loss_list_g), ", Loss D:", np.mean(loss_list_d))