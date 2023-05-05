import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from config import config

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.Tanh1 = nn.Tanh()
        self.Tanh2 = nn.Tanh()
        self.Softmax = nn.Softmax(dim=1)

        self.d_conv1 = nn.Conv2d(2, 24, 3, dilation=1, padding=1)
        self.d_conv2 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv3 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv4 = nn.Conv2d(24, 1, 3, dilation=1, padding=1)

        self.d_bn1 = nn.BatchNorm2d(24)
        self.d_bn2 = nn.BatchNorm2d(24)
        self.d_bn3 = nn.BatchNorm2d(24)
        self.d_bn4 = nn.BatchNorm2d(1)
        self.d_bn5 = nn.BatchNorm2d(128)
        self.d_bn6 = nn.BatchNorm2d(64)
        self.d_bn7 = nn.BatchNorm2d(3)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

        self.init_weights()

    def forward(self, input_images):  # 输入[3B, 2, 128, 128],输出[B, 3] * 3
        # net = F.max_pool2d(input_images, kernel_size=[2, 2])  # [3B, 2, 64, 64]
        # net = F.max_pool2d(net, kernel_size=[2, 2])  # [3B, 2, 32, 32]
        net = F.max_pool2d(input_images, kernel_size=[input_images.size(-2) // 32, input_images.size(-1) // 32])

        net = self.d_conv1(net)
        net = self.d_bn1(net)
        net = self.leakyrelu1(net)

        net = self.d_conv2(net)
        net = self.d_bn2(net)
        net = self.leakyrelu2(net)

        net = self.d_conv3(net)
        net = self.d_bn3(net)
        net = self.leakyrelu3(net)

        net = self.d_conv4(net)
        net = self.d_bn4(net)
        net1 = self.leakyrelu4(net)  # [3B, 1, 32, 32]

        net = net1.view(-1, 1024)  # [3B, 1024]
        net = self.fc1(net)  # [3B, 128]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn5(net)
        net = self.Tanh1(net)  # [3B, 128, 1, 1]

        net = net.view(-1, 128)  # [3B, 128]
        net = self.fc2(net)  # [3B, 64]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn6(net)
        net = self.Tanh2(net)  # [3B, 64, 1, 1]

        net = net.view(-1, 64)  # [3B, 64]
        net = self.fc3(net)  # [3B, 3]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn7(net)
        net = self.Softmax(net)  # [3B, 3, 1, 1]
        net = net.squeeze(3).squeeze(2)

        # realscore0, realscore1, realscore2 = torch.split(net, config.mini_batch_size, dim=0)
        realscore0, realscore1, realscore2 = torch.split(net, net.size(0)//3, dim=0)
        # feat0, feat1, feat2 = torch.split(net1, config.mini_batch_size, dim=0)
        feat0, feat1, feat2 = torch.split(net1, net.size(0)//3, dim=0)
        featDist = torch.mean(torch.pow(feat1 - feat2, 2))

        return realscore0, realscore1, realscore2, featDist

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        print('dis init_weights has been done!')

if __name__ == '__main__':
    # mini_batch_size = 20
    input = torch.randn(192, 2, 224, 224)
    model = discriminator()
    print(model)
    logits_real, logits_fake1, logits_fake2, Lgc = model(input)
    print('out:')
    print(logits_real.shape, logits_fake1.shape, logits_fake2.shape, Lgc.shape)
