import argparse
import os
import numpy as np
import itertools

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch

from load_compas_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000,
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
parser.add_argument("--latent_dim", type=int, default=10,
                    help="dimensionality of the latent code")

opt = parser.parse_args()
print(opt)

X, y, x_control = load_compas_data()

X = np.c_[X, x_control['race']]

X = torch.from_numpy(X)
x_control = torch.from_numpy(x_control['race'])
y = torch.from_numpy(y)

input_shape = X.shape

cuda = True if torch.cuda.is_available() else False


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(
        Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(8, opt.latent_dim)
        self.logvar = nn.Linear(8, opt.latent_dim)

    def forward(self, input):
        x = self.model(input)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=input_shape),
            nn.Linear(8, int(np.prod(input_shape))),
            nn.Tanh(),
        )

    def forward(self, input):
        g = self.model(input)
        return g


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        validity = self.model(input)
        return validity


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
l1_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    l1_loss.cuda()

# Configure data loader
dataloader = DataLoader(
    data,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (examples, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(examples.shape[0], 1).fill_(
            1.0), requires_grad=False)
        fake = Variable(Tensor(examples.shape[0], 1).fill_(
            0.0), requires_grad=False)

        # Configure input
        real_examples = Variable(examples.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_examples = encoder(real_examples)
        decoded_examples = decoder(encoded_examples)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(encoded_examples), valid) + 0.999 * l1_loss(
            decoded_examples, real_examples
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(
            0, 1, (examples.shape[0], opt.latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(
            discriminator(encoded_examples.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
