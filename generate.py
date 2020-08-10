import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import helpers as H
from generator import SGANGenerator

args = H.get_generation_arguments()

H.DEVICE = torch.device('cpu')

gen = SGANGenerator('generic_generator', args.sgan_layers, args.input_channels).to(H.DEVICE)
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

if args.tiling_rows > 0 or args.tiling_columns > 0:
    best=1e6
    crop1=0
    crop2=0
    for i in range(2 ** args.sgan_layers, 2 ** (args.sgan_layers + 1)):
        for j in range(2 ** args.sgan_layers, 2 ** (args.sgan_layers + 1)):
            loss = (torch.abs(texture[:, :, :, i] - texture[:, :, :, -j]).mean() if args.tiling_columns > 0 else 0) \
                 + (torch.abs(texture[:, :, i] - texture[:, :, -j]).mean().item() if args.tiling_rows > 0 else 0)
            if loss < best:
                best=loss
                crop1=i
                crop2=j
    #print(crop1, crop2, best)

if args.tiling_rows > 0:
    texture = texture[:, :, crop1:-crop2, :]

if args.tiling_columns > 0:
    texture = texture[:, :, :, crop1:-crop2]

if args.tiling_rows > 1:
    for i in range(args.tiling_rows - 1):
        texture = torch.cat((texture, texture), 2)

if args.tiling_columns > 1:
    for j in range(args.tiling_columns - 1):
        texture = torch.cat((texture, texture), 3)

texture = texture[0, :, :, :].numpy().T

if args.output_path:
    img = Image.fromarray(np.uint8(texture * 255.0))
    img.save(os.path.join(args.output_path, os.path.basename(args.saved_model_path)) + '_output.jpg')
else:
    plt.title('Texture image generated with SGAN' + str(args.sgan_layers) + ':')
    plt.imshow(texture)
    plt.show()
