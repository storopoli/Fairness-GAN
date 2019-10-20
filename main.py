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
parser.add_argument("--n_epochs", type=int, default=1000,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--p_dropout", type=float, default=0.2,
                    help="probability of activation unit dropout in layers")
parser.add_argument("--n_cpu", type=int, default=4,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=8,
                    help="dimensionality of the latent code")
parser.add_argument("--alpha_value", type=float, default=1,
                    help="alpha regularizer for classification utility")
parser.add_argument("--beta_value", type=float, default=1,
                    help="beta regularizer for decoder reconstruction of the inputs")
parser.add_argument("--gamma_value", type=float, default=1,
                    help="gamma regularizer for fair classification")

opt = parser.parse_args()
print(opt)

torch.manual_seed(1234)  # for reproducibility

X, Y, Z = load_compas_data()


ds = np.c_[X, Z['race'], Y]

# Y is "two_year_recid"
# X are ['age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25', 'race', 'sex', 'priors_count', 'c_charge_degree']
# Z is 'race' 0 White; 1 Black

cuda = True if torch.cuda.is_available() else False


class DatasetCompas(Dataset):
    def __init__(self, ds):
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index, 0:-1]  # all expect last column Y
        Z = self.data[index, -2]  # only penultimate column Z
        Y = self.data[index, -1]  # only last column Y
        sample = {'X': X, 'Z': Z, 'Y': Y}
        return sample


train_dataset = DatasetCompas(ds)

dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True
)

print('# training samples:', len(train_dataset))
print('# batches:', len(dataloader))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.p_dropout),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.p_dropout),
            nn.Linear(8, opt.latent_dim)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim + 1, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.p_dropout),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.p_dropout),
            nn.Linear(8, 8)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.p_dropout),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.p_dropout),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)
        return y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.p_dropout),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.p_dropout),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.model(x)
        return z


# Loss functions
BCE_loss = torch.nn.BCELoss()
MSE_loss = torch.nn.MSELoss()
CrossEntropy_loss = torch.nn.CrossEntropyLoss()

# Initialize networks
encoder = Encoder()
decoder = Decoder()
classifier = Classifier()
discriminator = Discriminator()


# CUDA
if cuda:
    encoder.cuda()
    decoder.cuda()
    classifier.cuda()
    discriminator.cuda()
    BCE_loss.cuda()
    MSE_loss.cuda()
    CrossEntropy_loss.cuda()

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Dec = torch.optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_C = torch.optim.Adam(
    classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Dis = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Hyperparameters Regularizer
alpha = Tensor([opt.alpha_value])
beta = Tensor([opt.beta_value])
gamma = Tensor([opt.gamma_value])


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        x = batch['X'].float()
        z = batch['Z'].float()
        y = batch['Y'].float()

        # adding an extra channel to Z and Y (N, M, 1)
        z = z.unsqueeze_(-1)
        y = y.unsqueeze_(-1)

        if cuda:
            x = x.cuda()
            z = z.cuda()
            y = y.cuda()


        x_tilde = encoder(x).detach()
        # -----------------
        #  Train Decoder
        # -----------------

        optimizer_Dec.zero_grad()

        # Reconstruct a batch of encoded examples
        x_hat = decoder(torch.cat((x_tilde, z), -1))

        # Loss measures decoder ability to reconstruct the generated examples
        dec_loss = (MSE_loss(x_hat, x) * beta)  # TODO: insert beta hyperparameters
        dec_loss.backward(retain_graph=True)
        optimizer_Dec.step()


        # ---------------------
        #  Train Classifier
        # ---------------------

        optimizer_C.zero_grad()

        # Classify a batch of examples
        y_hat = classifier(x_tilde)

        # Measure classifier's ability to classify real Y from generated samples' Y_hat
        cla_loss = (BCE_loss(y_hat, y) * alpha)  # TODO: insert alpha hyperparameters
        cla_loss.backward(retain_graph=True)
        optimizer_C.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_Dis.zero_grad()

        # Discriminate a batch of examples
        z_hat = discriminator(x_tilde)

        # Measure discriminator's ability to discrimante real Z from generated samples' Z_hat
        dis_loss = (BCE_loss(z_hat, z) * gamma)  # TODO: insert gamma hyperparameters

        dis_loss.backward(retain_graph=True)
        optimizer_Dis.step()


        # ---------------------
        #  Train Encoder
        # ---------------------

        optimizer_E.zero_grad()

        # Generate a batch of examples
        x_tilde = encoder(x)

        enc_loss = dec_loss + cla_loss - dis_loss

        enc_loss.backward()
        optimizer_E.step()


    # Time it
    end_time = time.time()
    time_taken = end_time - start_time

    print(
        "[Epoch %d/%d] [enc loss: %f] [dec loss: %f] [cla loss: %f] [dis loss: %f] [Time: %f]"
        % (epoch, opt.n_epochs, enc_loss.item(), dec_loss.item(), cla_loss.item(), dis_loss.item(), time_taken)
    )


torch.save({
    'Encoder': encoder.state_dict(),
    'Decoder': decoder.state_dict(),
    'Classifier': classifier.state_dict(),
    'Discriminator': discriminator.state_dict(),
    'optimizer_E': optimizer_E.state_dict(),
    'optimizer_Dec': optimizer_Dec.state_dict(),
    'optimizer_C': optimizer_C.state_dict(),
    'optimizer_Dis': optimizer_Dis.state_dict(),
}, f"./saved_models/compas-alpha_{opt.alpha_value}-beta_{opt.beta_value}-gamma_{opt.gamma_value}.pt")
