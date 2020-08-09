import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import helpers as H
from generator import SGANGenerator

args = H.get_generation_arguments()

gen = SGANGenerator(args.input_channels, args.sgan_layers).to(H.DEVICE)
gen.load_state_dict(torch.load(args.saved_model_path, map_location = H.DEVICE))
gen.eval()

input_height = args.input_size + (4 if args.tiling_rows > 0 else 0)
input_width = args.input_size + (4 if args.tiling_columns > 0 else 0)
input_tensor = torch.rand(1, args.input_channels, input_height, input_width).to(H.DEVICE) * 2.0 - 1.0

if args.tiling_rows > 0:
    input_tensor[:, :, :4, :] = input_tensor[:, :, -4:, :]

if args.tiling_columns > 0:
    input_tensor[:, :, :, :4] = input_tensor[:, :, :, -4:]

with torch.no_grad():
    texture = gen(input_tensor).cpu() / 2.0 + 0.5

if args.tiling_rows > 0:
    crop_size = 2 ** (args.sgan_layers + 2)
    crop_row = 2 ** args.sgan_layers
    best_crop_row = None
    best_error = inf

    for k in range(crop_row):
        curr_error = torch.mean(torch.square(texture[:, :, crop_row + k, :] - texture[:, :, crop_row + k + crop_size, :]))

        if curr_error < best_error:
            best_error = curr_error
            best_crop_row = crop_row + k

    texture = texture[:, :, best_crop_row:best_crop_row + crop_size, :]

if args.tiling_columns > 0:
    crop_size = 2 ** (args.sgan_layers + 2)
    crop_column = 2 ** args.sgan_layers
    best_crop_column = None
    best_error = inf

    for k in range(crop_column):
        curr_error = torch.mean(torch.square(texture[:, :, :, crop_column + k] - texture[:, :, :, crop_column + k + crop_size]))

        if curr_error < best_error:
            best_error = curr_error
            best_crop_column = crop_column + k

    texture = texture[:, :, :, best_crop_column:best_crop_column + crop_size]

if args.tiling_rows > 1:
    for i in range(args.tiling_rows - 1):
        texture = torch.cat((texture, texture), 2)

if args.tiling_columns > 1:
    for j in range(args.tiling_columns - 1):
        texture = torch.cat((texture, texture), 3)

texture = texture[0, :, :, :].numpy().T

plt.title('Texture image generated with SGAN' + str(args.sgan_layers) + ':')
plt.imshow(texture)
plt.show()

if args.output_path:
    img = PIL.fromarray(np.uint8(texture * 255.0))
    img.save(os.path.join(args.output_path, os.path.splitext(os.path.basename(args.saved_model_path)) + '_output.jpg'))
