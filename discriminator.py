import torch
import torch.nn as nn
import torch.nn.functional as F

class SGANDiscrimantor(nn.Module):
    def __init__(self, num_layers, lr = 2e-4):
        super(SGANDiscrimantor, self).__init__()

        NUM_FILTERS = [2 ** (i + 6) for i in range(num_layers - 1)] + [1]
        KERNEL_SIZE = 5
        STRIDE = 2
        PADDING = KERNEL_SIZE // 2

        self.name = 'SGAN' + str(num_layers) + '_DIS'
        self.layers = nn.ModuleList()

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

        self.optim = torch.optim.Adam(self.parameters(), lr = lr, betas = (0.5, 0.999), weight_decay = 1e-5)
        self.loss_history = []

    def loss(self, pred_real, pred_fake):
        loss = torch.mean(F.binary_cross_entropy_with_logits(pred_fake, pred_fake.new_zeros(pred_fake.size()))) \
            + torch.mean(F.binary_cross_entropy_with_logits(pred_real, pred_real.new_ones(pred_real.size())))
        self.loss_history.append(loss.item())

        return loss

    def forward(self, input_tensor):
        output = input_tensor

        for layer in self.layers:
            output = layer(output)

        return output

    def training_step(self, img_real, img_fake):
        self.zero_grad()
        pred_fake = self(img_fake.detach())
        pred_real = self(img_real)
        loss = self.loss(pred_real, pred_fake)
        loss.backward()
        self.optim.step()
        self.zero_grad()
