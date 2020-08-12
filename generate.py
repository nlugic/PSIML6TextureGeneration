import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import helpers as H
from generator import SGANGenerator

args = H.get_generation_arguments()

H.DEVICE = torch.device('cpu')

state_dict = torch.load(args.saved_model_path, map_location = H.DEVICE)
input_channels = state_dict['layers.0.weight'].shape[0]
gen = SGANGenerator('generic_generator', args.sgan_layers, input_channels).to(H.DEVICE)
gen.load_state_dict(state_dict)
gen.eval()

input_height = args.input_size + (4 if args.tiling_rows > 0 else 0)
input_width = args.input_size + (4 if args.tiling_columns > 0 else 0)
input_tensor = torch.rand(1, input_channels, input_width, input_height).to(H.DEVICE) * 2.0 - 1.0

if args.tiling_rows > 0:
    input_tensor[:, :, :, :4] = input_tensor[:, :, :, -4:]

if args.tiling_columns > 0:
    input_tensor[:, :, :4, :] = input_tensor[:, :, -4:, :]

with torch.no_grad():
    texture = gen(input_tensor).cpu() / 2.0 + 0.5

if args.tiling_rows > 0 or args.tiling_columns > 0:
    best_loss = float('inf')
    crop_left_top = 0
    crop_right_bottom = 0

    for i in range(2 ** args.sgan_layers, 2 ** (args.sgan_layers + 1)):
        for j in range(2 ** args.sgan_layers, 2 ** (args.sgan_layers + 1)):
            loss = (torch.mean(torch.abs(texture[:, :, i, :] - texture[:, :, -j, :])) if args.tiling_columns > 0 else 0) \
                 + (torch.mean(torch.abs(texture[:, :, :, i] - texture[:, :, :, -j])) if args.tiling_rows > 0 else 0)

            if loss < best_loss:
                best_loss = loss
                crop_left_top = i
                crop_right_bottom = j

    if args.tiling_rows > 0:
        texture = texture[:, :, :, crop_left_top:-crop_right_bottom]

    if args.tiling_columns > 0:
        texture = texture[:, :, crop_left_top:-crop_right_bottom, :]

if args.tiling_rows > 1:
    initial_texture = texture

    for i in range(args.tiling_rows - 1):
        texture = torch.cat((texture, initial_texture), 3)

if args.tiling_columns > 1:
    initial_texture = texture

    for j in range(args.tiling_columns - 1):
        texture = torch.cat((texture, initial_texture), 2)

texture = texture[0, :, :, :].numpy().T

if args.output_path:
    img = Image.fromarray(np.uint8(texture * 255.0)).rotate(90.0)
    img.save(os.path.join(args.output_path, os.path.basename(args.saved_model_path)) + '_output.jpg')
else:
    plt.title('Texture image generated with SGAN' + str(args.sgan_layers) + ':')
    plt.imshow(texture)
    plt.show()
