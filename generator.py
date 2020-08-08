import torch
import torch.nn as nn

class SGANGenerator(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(SGANGenerator, self).__init__()
        self.name = 'SGAN' + str(num_layers) + '_GEN'
        self.layers = nn.ModuleList()
        
        NUM_FILTERS = [2 ** (num_layers + 4 - i) for i in range(num_layers - 1)] + [3] # 3 -> RGB output
        KERNEL_SIZE = 5
        STRIDE = 2
        PADDING = KERNEL_SIZE // 2

        for i in range(num_layers):
            self.layers.append(nn.ConvTranspose2d(
                in_channels if i == 0 else NUM_FILTERS[i - 1],
                NUM_FILTERS[i],
                kernel_size = KERNEL_SIZE,
                stride = STRIDE,
                padding = PADDING,
                #output_padding=1
            ))

            if i < num_layers - 1:
                self.layers.append(nn.BatchNorm2d(NUM_FILTERS[i], eps = 1e-04, momentum = 0.1))
                self.layers.append(nn.ReLU(inplace = True))
            else:
                # za poslednji sloj rade tanh bez batch norma
                self.layers.append(nn.Tanh())

    def forward(self, input_tensor):
        output = input_tensor
        
        for layer in self.layers:
            output = layer(output)
        
        return output
