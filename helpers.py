import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# treba da se dodaju stvari vezane za device, tako da kad se slike ucitavaju budu odma na GPU

# uradi se model.apply(init_sgan_weights) da bi se sve tezine u mrezi inicijalizovale
def init_sgan_weights(node):
    if isinstance(node, nn.Conv2d) or isinstance(node, nn.ConvTranspose2d):
        nn.init.normal_(node.weight.data, std = 0.02)
        nn.init.normal_(node.bias.data, std = 0.02) # ovo je u kodu najverovatnije 0.0
    elif isinstance(node, nn.BatchNorm2d):
        #nn.init.normal_(node.weight.data, std = 0.02) # ovo je u kodu najverovatnije sa mean = 1.0
        nn.init.normal_(node.weight.data, mean=1.0, std = 0.02)
        #nn.init.normal_(node.bias.data, std = 0.02) # ovo je u kodu najverovatnije 0.0

def image_to_tensor(img):
    tensor = torch.Tensor(np.array(img).transpose((2, 0, 1)))
    return tensor / 255.0 * 2.0 - 1.0 # u kodu je -1..1 pa sam i ovde tako stavio, mozda treba od 0..1 

def save_tensor_as_image(tensor, filename):
    img = tensor.numpy().transpose((1, 2, 0)) # treba videti da li treba ovde i gore transpose
    img = np.uint8((img + 1.0) * 255.0 / 2.0)

    Image.fromarray(img).save(filename)

def get_train_dataset(path, device, size = 128, batch_size = 64, mirror = True):
    images_to_sample = []

    for file in os.listdir(path): # promeniti tako da pita dal je fajl
        name = path + file
        try:
            img = Image.open(name)
            images_to_sample += [image_to_tensor(img).to(device)]
            if mirror:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                images_to_sample += [image_to_tensor(img).to(device)]
        except:
            print("Image ", name, " failed to load!")

    while True:
        train_dataset = torch.zeros(batch_size, 3, size, size)
        for i in range(batch_size):
            image_to_sample = images_to_sample[np.random.randint(len(images_to_sample))]
            if size < image_to_sample.shape[1] and size < image_to_sample.shape[2]:
                h = np.random.randint(image_to_sample.shape[1] - size)
                w = np.random.randint(image_to_sample.shape[2] - size)
                img = image_to_sample[:, h:h + size, w:w + size]
            else:
                img = image_to_sample

            train_dataset[i, :, :, :] = img

        yield train_dataset
