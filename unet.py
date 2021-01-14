import torch
import torch.nn as nn

def conv_1(in_c, out_c, bt):
    if bt:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1 ,bias= False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1 ,bias= False),
            nn.LeakyReLU(0.2, inplace=True)
        )
    return conv

def de_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1 ,bias= False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )
    return conv

class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        self.conv1 = conv_1(opt.nc, opt.nef, False)
        # state size: (nef) x 64 x 64
        self.conv2 = conv_1(opt.nef,opt.nef, True)
        # state size: (nef) x 32 x 32
        self.conv3 = conv_1(opt.nef, opt.nef*2, True)
        # state size: (nef*2) x 16 x 16
        self.conv4 = conv_1(opt.nef*2, opt.nef*4, True)
        # state size: (nef*4) x 8 x 8
        self.conv5 = conv_1(opt.nef*4, opt.nef*8, True)
        # state size: (nef*8) x 4 x 4
        self.conv6 = nn.Conv2d(opt.nef*8,opt.nBottleneck,4, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(opt.nBottleneck)
        self.leak_relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.convt1 = nn.ConvTranspose2d(opt.nBottleneck, opt.ngf * 8, 4, 1, 0, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(opt.ngf * 8)
        self.relu = nn.ReLU(True)
        # state size. (ngf*8) x 4 x 4
        self.convt2 = de_conv(opt.ngf * 16, opt.ngf * 4)
        # state size. (ngf*4) x 8 x 8
        self.convt3 = de_conv(opt.ngf * 8, opt.ngf * 2)
        # state size. (ngf*2) x 16 x 16
        self.convt4 = de_conv(opt.ngf * 4, opt.ngf)
        # state size. (ngf) x 32 x 32
        self.convt5 = de_conv(opt.ngf*2, opt.ngf)
        # state size. (ngf) x 64 x 64
        self.convt6 = nn.ConvTranspose2d(opt.ngf*2, opt.nc, 4, 2, 1, bias=False)
        self.tan = nn.Tanh()
        # state size. (nc) x 128 x 128
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.batchNorm1(x6)
        x8 = self.leak_relu(x7)
        
        x9 = self.convt1(x8)
        x10 = self.batchNorm2(x9)
        x11 = self.relu(x10)
        x12 = self.convt2(torch.cat([x5, x11], 1))
        x13 = self.convt3(torch.cat([x4, x12], 1))
        x14 = self.convt4(torch.cat([x3, x13], 1))
        x15 = self.convt5(torch.cat([x2, x14], 1))
        
        x16 = self.convt6(torch.cat([x1, x15], 1))
        
        return self.tan(x16)
        
        
        
        
#     def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#         return output


class _netlocalD(nn.Module):
    def __init__(self, opt):
        super(_netlocalD, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

    
class _netgloballD(nn.Module):
    def __init__(self, opt):
        super(_netgloballD, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 8, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)
