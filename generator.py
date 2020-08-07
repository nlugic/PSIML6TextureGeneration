import torch
import torch.nn as nn

class SGANGenerator(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(SGANGenerator, self).__init__()
        self.name = 'SGAN' + str(num_layers) + '_GEN'
        self.layers = []
        
        NUM_FILTERS = [2 ** (num_layers - i) for i in range(num_layers)]
        KERNEL_SIZE = 5
        STRIDE = 2
        PADDING = KERNEL_SIZE // 2

        for i in range(num_layers):
            # pitanje paddinga
            self.layers.append(nn.ConvTranspose2d(in_channels, NUM_FILTERS[i]), kernel_size = KERNEL_SIZE, stride = STRIDE, padding = PADDING)
            if i < num_layers - 1:
                # inplace mozda ne bude radio
                self.layers.append(nn.ReLU(inplace = True))
                # fale gamma i beta
                self.layers.append(nn.BatchNorm2d(NUM_FILTERS[i], eps = 1e-04, momentum = 0.1))
            else:
                # za poslednji sloj rade tanh bez batch norma
                self.layers.append(nn.Tanh())

    def forward(self, input_tensor):
        output = input_tensor
        
        for layer in self.layers:
            output = layer(output)
        
        return output