import torch
import torch.nn as nn

class SGANDiscrimantor(nn.Module):
    def __init__(self, model_name, num_layers):
        super(SGANDiscrimantor, self).__init__()
        self.name = 'SGAN' + str(num_layers) + '_' + model_name + '_DIS'
        self.layers = nn.ModuleList()

        NUM_FILTERS = [2 ** (i + 6) for i in range(num_layers - 1)] + [1]
        KERNEL_SIZE = 5
        STRIDE = 2
        PADDING = KERNEL_SIZE // 2

        for i in range(num_layers):
            self.layers.append(nn.Conv2d(
                3 if i == 0 else NUM_FILTERS[i - 1],
                NUM_FILTERS[i],
                kernel_size = KERNEL_SIZE,
                stride = STRIDE,
                padding = PADDING
            ))

            if i < num_layers - 1:
                if i > 0:
                    self.layers.append(nn.BatchNorm2d(NUM_FILTERS[i], eps = 1e-4, momentum = 0.1))

                self.layers.append(nn.LeakyReLU(0.2, inplace = True))

    def forward(self, input_tensor):
        output = input_tensor

        for layer in self.layers:
            output = layer(output)

        return output
