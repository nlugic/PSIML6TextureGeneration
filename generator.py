import torch
import torch.nn as nn
import torch.nn.functional as F

class SGANGenerator(nn.Module):
    def __init__(self, num_layers, in_channels, lr = 2e-4):
        super(SGANGenerator, self).__init__()

        NUM_FILTERS = [2 ** (num_layers + 4 - i) for i in range(num_layers - 1)] + [3]
        KERNEL_SIZE = 5
        STRIDE = 2
        PADDING = KERNEL_SIZE // 2

        self.name = 'SGAN' + str(num_layers) + '_GEN'
        self.layers = nn.ModuleList()

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

        self.optim = torch.optim.Adam(self.parameters(), lr = lr, betas = (0.5, 0.999), weight_decay = 1e-5)
        self.loss_history = []

    def loss(self, pred_fake):
        loss = torch.mean(F.binary_cross_entropy_with_logits(pred_fake, pred_fake.new_ones(pred_fake.size())))
        self.loss_history.append(loss.item())

        return loss

    def forward(self, input_tensor):
        output = input_tensor

        for layer in self.layers:
            output = layer(output)

        return output

    def training_step(self, pred_fake):
        self.zero_grad()
        loss = self.loss(pred_fake)
        loss.backward()
        self.optim.step()
        self.zero_grad()
