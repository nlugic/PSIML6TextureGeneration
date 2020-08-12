import argparse, os, random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_training_arguments():
    arg_parser = argparse.ArgumentParser(description = 'Train the SGAN model on a picture/set of pictures.')
    arg_parser.add_argument('dataset_path', metavar = 'TRAIN_DATA_PATH', type = str, help = 'Path to the image file/folder containing images which will be used as training data.')
    arg_parser.add_argument('model_name', metavar = 'MODEL_NAME', type = str, help = 'A name under which the SGAN model will be trained and saved.')
    arg_parser.add_argument('--sgan_layers', '-l', metavar = 'SGAN_LAYERS', nargs = '?', const = 5, default = 5, type = int, help = 'Number of Conv2d/ConvTranspose2d layers in the model\'s discriminator/generator.')
    arg_parser.add_argument('--input_size', '-s', metavar = 'INPUT_SIZE', nargs = '?', const = 10, default = 10, type = int, help = 'Height/width of the model\'s input.')
    arg_parser.add_argument('--input_channels', '-c', metavar = 'INPUT_CHANNELS', nargs = '?', const = 50, default = 50, type = int, help = 'Number of channels in the model\'s input.')
    arg_parser.add_argument('--training_epochs', '-e', metavar = 'TRAIN_NUMBER_OF_EPOCHS', nargs = '?', const = 45, default = 45, type = int, help = 'Number of epochs used for training.')
    arg_parser.add_argument('--batch_size', '-b', metavar = 'TRAIN_EXAMPLES_PER_BATCH', nargs = '?', const = 32, default = 32, type = int, help = 'Minibatch size for training the SGAN model.')
    arg_parser.add_argument('--learning_rate', '-lr', metavar = 'MODEL_LEARNING_RATE', nargs = '?', const = 2e-4, default = 2e-4, type = float, help = 'Learning rate used for training the SGAN model.')
    arg_parser.add_argument('--output_path', '-o', metavar = 'MODEL_OUTPUT_PATH', nargs = '?', const = './', default = './', type = str, help = 'Directory to which the trained SGAN model should be saved.')
    arg_parser.add_argument('--cuda_device', '-d', metavar = 'TRAIN_CUDA_DEVICE', nargs = '?', const = 0, default = 0, type = int, help = 'ID of CUDA device which should be used for training.')

    return arg_parser.parse_args()

def get_generation_arguments():
    arg_parser = argparse.ArgumentParser(description = 'Use a pretrained SGAN model to generate a texture.')
    arg_parser.add_argument('saved_model_path', metavar = 'SAVED_MODEL_PATH', type = str, help = 'Path to the saved SGAN model.')
    arg_parser.add_argument('--sgan_layers', '-l', metavar = 'SGAN_LAYERS', nargs = '?', const = 5, default = 5, type = int, help = 'Number of Conv2d/ConvTranspose2d layers in the model\'s discriminator/generator.')
    arg_parser.add_argument('--input_size', '-s', metavar = 'INPUT_SIZE', nargs = '?', const = 20, default = 20, type = int, help = 'Height/width of the model\'s input.')
    arg_parser.add_argument('--tiling_rows', '-r', metavar = 'TILING_ROWS', nargs = '?', const = 0, default = 0, type = int, help = 'Number of rows for tiling texture generation.')
    arg_parser.add_argument('--tiling_columns', '-c', metavar = 'TILING_COLUMNS', nargs = '?', const = 0, default = 0, type = int, help = 'Number of columns for tiling texture generation.')
    arg_parser.add_argument('--output_path', '-o', metavar = 'OUTPUT_PATH', nargs = '?', const = '', type = str, help = 'Path to the output directory.')

    return arg_parser.parse_args()

def init_sgan_weights(node):
    if isinstance(node, nn.Conv2d) or isinstance(node, nn.ConvTranspose2d):
        nn.init.normal_(node.weight.data, std = 0.02)
        nn.init.constant_(node.bias.data, 0.0)
    elif isinstance(node, nn.BatchNorm2d):
        nn.init.normal_(node.weight.data, mean = 1.0, std = 0.02)
        nn.init.constant_(node.bias.data, 0.0)

def image_to_tensor(img):
    tensor = torch.Tensor(np.array(img).transpose((2, 0, 1)))
    return tensor / 255.0 * 2.0 - 1.0

def save_tensor_as_image(tensor, filename):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = np.uint8((img + 1.0) * 255.0 / 2.0)

    Image.fromarray(img).save(filename)

def get_train_dataset(path, device, size, batch_size = 64, mirror = True):
    images_to_sample = []

    if os.path.isdir(path):
        for file in os.listdir(path):
            filename = os.path.join(path, file)
            try:
                img = Image.open(filename)
                images_to_sample.append(image_to_tensor(img).to(DEVICE))
                if mirror:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    images_to_sample.append(image_to_tensor(img).to(DEVICE))
            except:
                print("Image ", filename, " failed to load!")

        random.shuffle(images_to_sample)
    else:
        images_to_sample.append(image_to_tensor(Image.open(path)))

    train_dataset = torch.zeros(batch_size, 3, size, size, device = DEVICE)

    while True:
        for i in range(batch_size):
            image_to_sample = images_to_sample[i % len(images_to_sample)]

            if size < image_to_sample.shape[1] and size < image_to_sample.shape[2]:
                h = np.random.randint(image_to_sample.shape[1] - size)
                w = np.random.randint(image_to_sample.shape[2] - size)
                img = image_to_sample[:, h:h + size, w:w + size]
            else:
                img = image_to_sample

            train_dataset[i, :, :, :] = img

        yield train_dataset
