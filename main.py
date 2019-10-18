#!/usr/bin/env python
import argparse
import numpy as np
import time

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn
import torch

from load_compas_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=9,
                    help="dimensionality of the latent code")

opt = parser.parse_args()
print(opt)

torch.manual_seed(1234)  # for reproducibility

X, Y, A = load_compas_data()


ds = np.c_[X, A['race'], Y]

# Y is "two_year_recid"
# X are ['age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25', 'race', 'sex', 'priors_count', 'c_charge_degree']
# A is 'race' 0 White; 1 Black


input_shape = ds.shape[1]

cuda = True if torch.cuda.is_available() else False


class DatasetCompas(Dataset):
    def __init__(self, ds):
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index, 0:-1]
        sensible = self.data.iloc[index, -2]
        label = self.data.iloc[index, -1]
        return data, sensible, label


train_dataset = DatasetCompas(ds)

dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)


def reparameterization(mu, logvar):  # insert random gaussian noise into tensor
    std = torch.exp(logvar / 2)
    sampled_z = Variable(
        Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(9, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(9, input_shape)
        )

        self.mu = nn.Linear(9, opt.latent_dim)
        self.logvar = nn.Linear(9, opt.latent_dim)

    def forward(self, x):
        x = self.model(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(9, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(9, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        y = self.model(z)
        return y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(9, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(9, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        a = self.model(x)
        return a


# Loss functions
BCE_loss = torch.nn.BCELoss()
MSE_loss = torch.nn.MSELoss()

# Initialize generator and discriminators
generator = Generator()
classifier = Classifier()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator_1.cuda()
    discriminator_2.cuda()
    BCE_loss.cuda()
    MSE_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_C = torch.optim.Adam(
    classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    start_time = time.time()
    for i, batch in enumerate(dataloader):

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (batch.shape[0], opt.latent_dim))))

        # Generate a batch of examples
        gen_examples = generator(z)

        # Loss measures generator's ability to fool the discriminator_1
        g_loss = MSE_loss(gen_examples, valid)

        # ---------------------
        #  Train Classifier
        # ---------------------

        optimizer_C.zero_grad()

        # Measure classifier's ability to classify real Y from generated samples' Y_hat
        c_loss = BCE_loss(Y, Y_hat)

        c_loss.backward()
        optimizer_C.step()

        # ---------------------
        #  Update Generator with error from G and C
        # ---------------------
        g_loss = g_loss + c_loss
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to discrimante real A from generated samples' A_hat
        d_loss = BCE_loss(A, A_hat)

        d_loss.backward()
        optimizer_D.step()

        # Time it
        end_time = time.time()
        time_taken = end_time - start_time

        print(
            "[Epoch %d/%d] [Batch %d/%d] [C loss: %f] [G loss: %f] [D loss: %f] [Time: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), c_loss.item(), g_loss.item(), d_loss.item(), time_taken)
        )

torch.save({
    'Generator': generator.state_dict(),
    'Classifier': classifier.state_dict(),
    'Discriminator': discriminator.state_dict(),
    'optimizer_G': optimizer_G.state_dict(),
    'optimizer_C': optimizer_C.state_dict(),
    'optimizer_D': optimizer_D.state_dict(),
}, './saved_models/compas.pt')
