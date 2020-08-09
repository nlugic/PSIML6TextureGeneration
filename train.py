import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import helpers as H
from generator import SGANGenerator
from discriminator import SGANDiscrimantor

args = H.get_training_arguments()
H.DEVICE = torch.device('cuda:' + str(args.cuda_device) if torch.cuda.is_available() else 'cpu')

gen = SGANGenerator(args.model_name, args.sgan_layers, args.input_channels).to(H.DEVICE)
gen.apply(H.init_sgan_weights)
dis = SGANDiscrimantor(args.model_name, args.sgan_layers).to(H.DEVICE)
dis.apply(H.init_sgan_weights)

dataset_iter = H.get_train_dataset(args.dataset_path, H.DEVICE, (args.input_size - 1) * 2 ** args.sgan_layers + 1, batch_size = args.batch_size)

loss_funct_g = lambda pred: torch.mean(F.binary_cross_entropy_with_logits(pred, pred.new_ones(pred.size())))
loss_funct_d = lambda pred_real, pred_fake: torch.mean(F.binary_cross_entropy_with_logits(pred_fake, pred_fake.new_zeros(pred_fake.size()))) + torch.mean(F.binary_cross_entropy_with_logits(pred_real, pred_real.new_ones(pred_real.size())))

optim_g = torch.optim.Adam(gen.parameters(), lr = args.learning_rate, betas = (0.5, 0.999), weight_decay = 1e-5)
optim_d = torch.optim.Adam(dis.parameters(), lr = args.learning_rate, betas = (0.5, 0.999), weight_decay = 1e-5)

writer = SummaryWriter()
z9 = torch.rand(args.batch_size, args.input_channels, args.input_size, args.input_size).to(H.DEVICE) * 2.0 - 1.0
z20 = torch.rand(args.batch_size, args.input_channels, 20, 20).to(H.DEVICE) * 2.0 - 1.0

curr_epoch = 0

while curr_epoch < args.training_epochs:
    print("Epoch", curr_epoch)
    curr_epoch += 1

    loss_list_d = []
    loss_list_g = []

    n_iter = 50 # oni imaju 100 iteracija, ali u jednoj treniraju ili gen ili dis, a mi oba
    for i, img_real in enumerate(tqdm(dataset_iter, total = n_iter)):
        if i >= n_iter:
            break

        z = torch.rand(args.batch_size, args.input_channels, args.input_size, args.input_size).to(H.DEVICE) * 2.0 - 1.0

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

    if curr_epoch % 15 == 0:
        torch.save(gen.state_dict(), os.path.join(args.output_path, gen.name + '_epoch_' + str(curr_epoch) + '.pt'))

    writer.add_scalar("Loss/Generator", np.mean(loss_list_g), curr_epoch)
    writer.add_scalar("Loss/Discriminator", np.mean(loss_list_d), curr_epoch)
    gen.eval()
    dis.eval()

    with torch.no_grad():
        writer.add_images("Generated images/l9", gen(z9).detach()[:4] / 2.0 + 0.5, curr_epoch)
        writer.add_images("Generated images/l20", gen(z20).detach()[:4] / 2.0 + 0.5, curr_epoch)

    gen.train()
    dis.train()
