from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from generator import SGANGenerator
from helpers import save_tensor_as_image

layers = 5
nz = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = SGANGenerator(nz, layers).to(device)
gen.load_state_dict(torch.load("gen-asphalt2k.pt", map_location=device)) # ucitava vec istreniran model
gen.eval()

z_tile = torch.rand(1, nz, 24, 24).to(device) * 2 - 1
z_tile[:,:,:4] = z_tile[:,:,-4:]
z_tile[:,:,:,:4] = z_tile[:,:,:,-4:]

with torch.no_grad():
    im_tile = gen(z_tile).cpu() / 2 + 0.5
print("geenrated")

crop1=34
crop2=63   
im_tile = im_tile[0, :, crop1:-crop2, crop1:-crop2]

plt.imshow( torch.cat((torch.cat((im_tile, im_tile),1), torch.cat((im_tile,im_tile),1)),2).cpu().numpy().T)
plt.show()