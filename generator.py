import torch
import torch.nn as nn

class SGANGenerator(nn.Module):
    def __init__(self, model_name, num_layers, in_channels):
        super(SGANGenerator, self).__init__()
        self.name = 'SGAN' + str(num_layers) + '_' + model_name + '_GEN'
        self.layers = nn.ModuleList()
        
        NUM_FILTERS = [2 ** (num_layers + 4 - i) for i in range(num_layers - 1)] + [3]
        KERNEL_SIZE = 5
        STRIDE = 2
        PADDING = KERNEL_SIZE // 2

        for i in range(num_layers):
            self.layers.append(nn.ConvTranspose2d(
                in_channels if i == 0 else NUM_FILTERS[i - 1],
                NUM_FILTERS[i],
                kernel_size = KERNEL_SIZE,
                stride = STRIDE,
                padding = PADDING
            ))

            if i < num_layers - 1:
                self.layers.append(nn.BatchNorm2d(NUM_FILTERS[i], eps = 1e-4, momentum = 0.1))
                self.layers.append(nn.ReLU(inplace = True))
            else:
                self.layers.append(nn.Tanh())

    def forward(self, input_tensor):
        output = input_tensor
        
        for layer in self.layers:
            output = layer(output)
        
        return output
