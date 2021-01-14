from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np

from unet import _netlocalD,_netG, _netgloballD
import utils

def main():
    print("Started reading the file")
    parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot',  default='data/', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--niter', type=int, default=40, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
    parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
    parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
    parser.add_argument('--wtl2',type=float,default=0.998,help='0 means do not use else use with this weight')
    parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs("result/train/cropped")
        os.makedirs("result/train/real")
        os.makedirs("result/train/recon")
        os.makedirs("model")
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    # elif opt.dataset == 'lsun':
    #     dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
    #                         transform=transforms.Compose([
    #                             transforms.Scale(opt.imageSize),
    #                             transforms.CenterCrop(opt.imageSize),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                         ]))
    # elif opt.dataset == 'cifar10':
    #     dataset = dset.CIFAR10(root=opt.dataroot, download=True,
    #                           transform=transforms.Compose([
    #                               transforms.Scale(opt.imageSize),
    #                               transforms.ToTensor(),
    #                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                           ])
    #     )
    # elif opt.dataset == 'streetview':
    #     transform = transforms.Compose([transforms.Scale(opt.imageSize),
    #                                     transforms.CenterCrop(opt.imageSize),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #     dataset = dset.ImageFolder(root=opt.dataroot, transform=transform )
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3
    nef = int(opt.nef)
    nBottleneck = int(opt.nBottleneck)
    wtl2 = float(opt.wtl2)
    overlapL2Weight = 10

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    resume_epoch=0

    netG = _netG(opt)
    netG.apply(weights_init)
#     if opt.netG != '':
#     print('loaded')
#     netG.load_state_dict(torch.load('model/netG_streetview2.pth',map_location=lambda storage, location: storage)['state_dict'])
#     resume_epoch = torch.load('model/netG_streetview2.pth')['epoch']
    print(netG)


    netD = _netlocalD(opt)
    netD.apply(weights_init)
#     if opt.netD != '':
    print('loaded')
#     netD.load_state_dict(torch.load('model/netlocalD2.pth',map_location=lambda storage, location: storage)['state_dict'])
#     resume_epoch = torch.load('model/netlocalD2.pth')['epoch']
    print(netD)
    
    netGD = _netgloballD(opt)
    netGD.apply(weights_init)
#     if opt.netD != '':
    print('loaded')
    path = ''
#     netGD.load_state_dict(torch.load('model/netlocalGD2.pth',map_location=lambda storage, location: storage)['state_dict'])
#     resume_epoch = torch.load('model/netlocalGD2.pth')['epoch']
#     print(netD)
    print(netGD)
    
    criterion = nn.BCELoss()
    criterionMSE = nn.MSELoss()

    input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize//2, opt.imageSize//2)

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        netGD.cuda()
        criterion.cuda()
        criterionMSE.cuda()
        input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()
        real_center = real_center.cuda()


    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    label = Variable(label)


    real_center = Variable(real_center)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))
    optimizerGD = optim.Adam(netGD.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))

    for epoch in range(resume_epoch,opt.niter):
        for i, data in enumerate(dataloader, 0):
            real_cpu, _ = data
#             data = [np.array(j) for j in data]
#             real_data = torch.Tensor(data)
            real_center_cpu = real_cpu[:,:,int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)]
            batch_size = real_cpu.size(0)
            with torch.no_grad():
                input_real.resize_(real_cpu.size()).copy_(real_cpu)
                input_cropped.resize_(real_cpu.size()).copy_(real_cpu)
                real_center.resize_(real_center_cpu.size()).copy_(real_center_cpu)
            input_cropped.data[:,0,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*117.0/255.0 - 1.0
            input_cropped.data[:,1,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*104.0/255.0 - 1.0
            input_cropped.data[:,2,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*123.0/255.0 - 1.0

            # train with real
            netD.zero_grad()
            netGD.zero_grad()
            with torch.no_grad():
                label.resize_(batch_size).fill_(real_label)

            output = netD(real_center)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()
            
            output_GD = netGD(input_real)
            errD_real_GD = criterion(output_GD, label)
            errD_real_GD.backward()
            D_x_GD = output_GD.data.mean()

            # train with fake
            # noise.data.resize_(batch_size, nz, 1, 1)
            # noise.data.normal_(0, 1)
            fake1 = netG(input_cropped)
            fake_size = fake1.size()
            fake = torch.cuda.FloatTensor(fake_size[0], fake_size[1], fake_size[2]//2, fake_size[3]//2)
            fake.data[:,0,:,:] = fake1.data[:,0, int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)]
            fake.data[:,1,:,:] = fake1.data[:,1, int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)]
            fake.data[:,2,:,:] = fake1.data[:,2, int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)]
            
            label.data.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            output_GD = netGD(fake1.detach())
            errD_fake_GD = criterion(output_GD, label)
            errD_fake_GD.backward()
            D_G_z1_GD = output_GD.data.mean()
            errD_GD = errD_real_GD + errD_fake_GD
#             if i%20==0 and i!=0:
            optimizerGD.step()


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG_D = criterion(output, label)

            output_GD = netGD(fake1)
            errG_D_global = criterion(output_GD, label)
            # errG_D.backward(retain_variables=True)

            # errG_l2 = criterionMSE(fake,real_center)
            wtl2Matrix = real_center.clone()
            wtl2Matrix.data.fill_(wtl2*overlapL2Weight)
            wtl2Matrix.data[:,:,int(opt.overlapPred):int(opt.imageSize/2 - opt.overlapPred),int(opt.overlapPred):int(opt.imageSize/2 - opt.overlapPred)] = wtl2

            errG_l2 = (fake-real_center).pow(2)
            errG_l2 = errG_l2 * wtl2Matrix
            errG_l2 = errG_l2.mean()
            
            errG_l3 = (fake1-input_real).pow(2)
            wtl3Matrix = input_real.clone()
            wtl3Matrix.data.fill_(wtl2)
            errG_l3 = wtl3Matrix*errG_l3
            errG_l3 = errG_l3.mean()
            
            errG = (0.00001) * errG_D +  0.1* errG_l2 + errG_l3 + (0.00001) *errG_D_global
#             if i%100==0 and i!=0: 
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f GLOB: %.4f'
                  % (epoch, opt.niter, i, len(dataloader), errD.data, errG_D.data,errG_l2.data, D_x,D_G_z1 ,errD_GD.data))
            if i % 200 == 0 and i!=0:
                vutils.save_image(real_cpu,
                        'result/real/real_samples_epoch_%03d.png' % (epoch))
                vutils.save_image(input_cropped.data,
                        'result/cropped/cropped_samples_epoch_%03d.png' % (epoch))
                recon_image = input_cropped.clone()
                recon_image.data[:,:,int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2)] = fake.data
                vutils.save_image(recon_image.data,
                        'result/recon/recon_center_samples_epoch_%03d.png' % (epoch))


        # do checkpointing
        torch.save({'epoch':epoch+1,
                    'state_dict':netG.state_dict()},
                    'model/netG_streetview'+str(epoch+1)+'.pth' )
        torch.save({'epoch':epoch+1,
                    'state_dict':netD.state_dict()},
                    'model/netlocalD'+str(epoch+1)+'.pth' )
        torch.save({'epoch':epoch+1,
                    'state_dict':netGD.state_dict()},
                    'model/netlocalGD'+str(epoch+1)+'.pth' )
if __name__ == '__main__':
    main()