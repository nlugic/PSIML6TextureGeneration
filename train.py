from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np

from discriminator import SGANDiscrimantor
from generator import SGANGenerator
from helpers import get_train_dataset, init_sgan_weights, save_tensor_as_image

layers = 5
random_size = 9
random_channels = 50
batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

gen = SGANGenerator(random_channels, layers).to(device)
gen.apply(init_sgan_weights)
dis = SGANDiscrimantor(layers).to(device)
dis.apply(init_sgan_weights)

dataset_folder = "train_textures/"
dataset_iter = get_train_dataset(dataset_folder, device, size = (random_size - 1) * 2 ** layers + 1, batch_size = batch_size)

loss_funct_g = lambda pred: torch.mean(F.binary_cross_entropy_with_logits(pred, pred.new_ones(pred.size())))
loss_funct_d = lambda pred_real, pred_fake: torch.mean(F.binary_cross_entropy_with_logits(pred_fake, pred_fake.new_zeros(pred_fake.size()))) + torch.mean(F.binary_cross_entropy_with_logits(pred_real, pred_real.new_ones(pred_real.size())))

optim_g = torch.optim.Adam(gen.parameters(), lr = 2e-4, betas = (0.5, 0.999), weight_decay = 1e-5)
optim_d = torch.optim.Adam(dis.parameters(), lr = 2e-4, betas = (0.5, 0.999), weight_decay = 1e-5)

writer = SummaryWriter()
z9 = torch.rand(batch_size, random_channels, 9, 9).to(device) * 2.0 - 1.0
z20 = torch.rand(batch_size, random_channels, 20, 20).to(device) * 2.0 - 1.0

curr_epoch = 0
total_epochs = 50

while curr_epoch < total_epochs:
    curr_epoch += 1
    print("Epoch", curr_epoch)

    loss_list_d = []
    loss_list_g = []

    n_iter = 50 # oni imaju 100 iteracija, ali u jednoj treniraju ili gen ili dis, a mi oba
    for i, img_real in enumerate(tqdm(dataset_iter, total = n_iter)):
        if i >= n_iter:
            break

        z = torch.rand(batch_size, random_channels, random_size, random_size).to(device) * 2.0 - 1.0

        # Train discriminator
        dis.zero_grad()
        img_fake = gen(z)
        pred_fake = dis(img_fake.detach())
        pred_real = dis(img_real)

        loss_d = loss_funct_d(pred_real, pred_fake)
        loss_list_d.append(loss_d.item())
        loss_d.backward()
        optim_d.step()
        
        # Train generator
        gen.zero_grad()
        pred_fake = dis(img_fake)

        loss_g = loss_funct_g(pred_fake)
        loss_list_g.append(loss_g.item())
        loss_g.backward()
        optim_g.step()

    writer.add_scalar("Loss/Generator", np.mean(loss_list_g), curr_epoch)
    writer.add_scalar("Loss/Discriminator", np.mean(loss_list_d), curr_epoch)
    gen.eval()
    dis.eval()
    with torch.no_grad():
        writer.add_images("Generated images/l9", gen(z9).detach()[:4] / 2 + 0.5, curr_epoch)
        writer.add_images("Generated images/l20", gen(z20).detach()[:4] / 2 + 0.5, curr_epoch)
    gen.train()
    dis.train()