import numpy as np
import os
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import torch.nn as nn
from tqdm import tqdm
from model import Generator, weights_init, Discriminator
from torch.utils.data import DataLoader
from data import HicDataset

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping
from ignite.engine import Engine

epochs = 100
HiC_max_value = 100
batch_size = 128

log_interval = 1000

def train(lowres, highres, val_lowres, val_hires, outModel, startmodel=None,startepoch=0, down_sample_ratio=16):
    low_resolution_samples = lowres.astype(np.float32) * down_sample_ratio
    high_resolution_samples = highres.astype(np.float32)

    val_lowres = val_lowres.astype(np.float32) * down_sample_ratio

    low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
    high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)

    val_lowres = np.minimum(HiC_max_value, val_lowres)
    val_hires = np.minimum(HiC_max_value, val_hires)

    netG = Generator()

    sample_size = low_resolution_samples.shape[-1]
    padding = 0
    half_padding = padding // 2
    output_length = sample_size - padding
    Y = []
    for i in range(high_resolution_samples.shape[0]):
        no_padding_sample = high_resolution_samples[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
        Y.append(no_padding_sample)
    Y = np.array(Y).astype(np.float32)
    Y = Y[:,np.newaxis,:,:]

    val_Y = []
    for i in range(val_hires.shape[0]):
        no_padding_sample = val_hires[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
        val_Y.append(no_padding_sample)
    val_Y = np.array(val_Y).astype(np.float32)
    val_Y = val_Y[:,np.newaxis,:,:]

    train_loader = DataLoader(HicDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(Y)), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(HicDataset(torch.from_numpy(val_lowres), torch.from_numpy(val_Y)), batch_size=batch_size, shuffle=True, drop_last=True)

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    netG.to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # if startmodel is not None:
    #     print('loading state dict from {}...'.format(startmodel))
    #     model.load_state_dict(torch.load(startmodel))
    #     print('finish loading state dict')

    # optimizer = optim.SGD(netG.parameters(), lr = 0.00001, momentum=0.9, weight_decay=0.0001)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(beta1, 0.999))

    real_label = 1.
    fake_label = 0.

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    progressBar = tqdm(range(epochs))

    for epoch in progressBar:
        for i, data in enumerate(train_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            progressBar.set_description('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (i+1) % 50 == 0:
                netG._save_to_state_dict(outModel+'.model')

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()