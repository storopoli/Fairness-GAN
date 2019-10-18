#!/usr/bin/env python
import argparse
import numpy as np
import itertools
import time

from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn as nn
import torch

from load_compas_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=8,
                    help="dimensionality of the latent code")

opt = parser.parse_args()
print(opt)

torch.manual_seed(1234)  # for reproducibility

X, Y, A = load_compas_data()

X = np.c_[X, A['race']]

X = torch.from_numpy(X)
A = torch.from_numpy(A['race'])
Y = torch.from_numpy(Y)

input_shape = X.shape[1]

cuda = True if torch.cuda.is_available() else False

if cuda:
    X = X.to(torch.device('cuda'))
    Y = Y.to(torch.device('cuda'))
    A = A.to(torch.device('cuda'))


dataloader = torch.utils.data.DataLoader(
    X,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)

# for i, batch in enumerate(dataloader):
#     print(i, batch)


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(
        Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, input_shape),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return z


class Discriminator_1(nn.Module):
    def __init__(self):
        super(Discriminator_1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class Discriminator_2(nn.Module):
    def __init__(self):
        super(Discriminator_2, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Other Loss functions
BCE_loss = torch.nn.BCELoss()
L1_loss = torch.nn.L1Loss()
MSE_loss = torch.nn.MSELoss()

# Initialize generator and discriminators
generator = Generator()
discriminator_1 = Discriminator_1()
discriminator_2 = Discriminator_2()

if cuda:
    generator.cuda()
    discriminator_1.cuda()
    discriminator_2.cuda()
    adversarial_loss.cuda()
    MSE_loss.cuda()
    L1_loss.cuda()
    BCE_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_1 = torch.optim.Adam(
    discriminator_1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_2 = torch.optim.Adam(
    discriminator_2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    start_time = time.time()
    for i, batch in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(batch.shape[0]).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch.shape[0]).fill_(0.0), requires_grad=False)

        # Configure input
        real_examples = Variable(batch.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (batch.shape[0], opt.latent_dim))))

        # Generate a batch of examples
        gen_examples = generator(z)

        # Loss measures generator's ability to fool the discriminator_1
        g_loss = adversarial_loss(
            discriminator_1(gen_examples), valid
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator_1
        # ---------------------

        optimizer_D_1.zero_grad()

        # Measure discriminator's 1 ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator_1(gen_examples), valid)
        fake_loss = adversarial_loss(discriminator_1(gen_examples.detach()), fake)
        d_1_loss = (real_loss + fake_loss) / 2

        d_1_loss.backward()
        optimizer_D_1.step()

        # Time it
        end_time = time.time()
        time_taken = end_time - start_time

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D_1 loss: %f] [G loss: %f] [Time: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_1_loss.item(), g_loss.item(), time_taken)
        )

torch.save({
    'Generator': generator.state_dict(),
    'Discriminator_1': discriminator_1.state_dict(),
    'Discriminator_2': discriminator_2.state_dict(),
    'optimizer_G': optimizer_G.state_dict(),
    'optimizer_D_1': optimizer_D_1.state_dict(),
    'optimizer_D_2': optimizer_D_2.state_dict(),
}, './saved_models/compas.pt')
