#!/usr/bin/env python
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import Accuracy

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
parser.add_argument("--p_dropout", type=float, default=0.2,
                    help="probability of activation unit dropout in layers")
parser.add_argument("--n_cpu", type=int, default=4,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=8,
                    help="dimensionality of the latent code")
parser.add_argument("--lambda_value", type=int, default=200,
                    help="lambda regularizer for fair classification")

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
        data = self.data[:, 0:-1]  # all expect last column Y
        sensible = self.data[:, -2]  # only penultimate column Z
        label = self.data[:, -1]  # only last column Y
        sample = {'data': data, 'sensible': sensible, 'label': label}
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
            nn.Linear(opt.latent_dim, 8),
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

# Initialize generator and discriminators
encoder = Encoder()
decoder = Decoder()
classifier = Classifier()
discriminator = Discriminator()



if cuda:
    encoder.cuda()
    decoder.cuda()
    classifier.cuda()
    discriminator.cuda()
    BCE_loss.cuda()
    MSE_loss.cuda()

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Dec = torch.optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_C = torch.optim.Adam(
    classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Dis = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Lambda
lambdas = Tensor([opt.lambda_value, 8])


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        x = batch['data'].float()
        z = batch['sensible'].float()
        y = batch['label'].float()

        # adding an extra channel to Z and Y (N, M, 1)
        z = z.unsqueeze_(-1)
        y = y.unsqueeze_(-1)

        if cuda:
            x = x.cuda()
            z = z.cuda()
            y = y.cuda()


        # -----------------
        #  Train Enconder/Decoder
        # -----------------

        optimizer_E.zero_grad()

        # Generate a batch of examples
        x_tilde = encoder(x)

        # Loss measures generator's ability to fool the discriminator_1
        enc_loss = MSE_loss(x_tilde, x)
        enc_loss.backward()
        optimizer_E.step()

        optimizer_Dec.zero_grad()

        # Reconstruct a batch of encoded examples
        x_tilde = encoder(x).detach()
        x_hat = decoder(x_tilde)

        # Loss measures generator's ability to fool the discriminator_1
        dec_loss = MSE_loss(x_hat, x)
        dec_loss.backward()
        optimizer_Dec.step()


        # ---------------------
        #  Train Classifier
        # ---------------------

        optimizer_C.zero_grad()

        # Discriminate a batch of examples
        z_hat = discriminator(x_tilde).detach()

        # Classify a batch of examples
        y_hat = classifier(x_tilde)

        # Measure classifier's ability to classify real Y from generated samples' Y_hat
        cla_loss = BCE_loss(y_hat, y) - (BCE_loss(z_hat, z) * lambdas).mean()
        cla_loss.backward()
        optimizer_C.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_Dis.zero_grad()

        # Classify a batch of examples
        y_hat = classifier(x_tilde).detach()

        # Discriminate a batch of examples
        z_hat = discriminator(x_tilde)


        # Measure discriminator's ability to discrimante real A from generated samples' A_hat
        dis_loss = (BCE_loss(z_hat, z) * lambdas).mean()

        dis_loss.backward()
        optimizer_Dis.step()


        # Time it
        end_time = time.time()
        time_taken = end_time - start_time

        print(
            "[Epoch %d/%d] [Batch %d/%d] [enc loss: %f] [dec loss: %f] [cla loss: %f] [dis loss: %f] [Time: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), enc_loss.item(), dec_loss.item(), cla_loss.item(), dis_loss.item(), time_taken)
        )

    if epoch % 100 == 0:  # for every 100 hundred epochs
        for i, batch in enumerate(dataloader):
            correct_cla = 0
            total_cla = 0
            correct_dis = 0
            total_dis = 0
            with torch.no_grad():
                Xs = batch['data'].float()
                Zs = batch['sensible'].float()
                Ys = batch['label'].float()

                # adding an extra channel to Z and Y (N, M, 1)
                Zs = Zs.unsqueeze_(-1)
                Ys = Ys.unsqueeze_(-1)

                if cuda:
                    Xs.cuda()
                    Zs.cuda()
                    Ys.cuda()
                ouputs_enc = encoder(Xs)
                outputs_cla = classifier(ouputs_enc)
                predicted_cla = (outputs_cla > 0.5).float()
                total_cla += Ys.size(0)
                correct_cla += (outputs_cla == Ys).sum().item()
                outputs_dis = discriminator(ouputs_enc)
                predicted_dis = (outputs_dis > 0.5).float()
                total_dis += Zs.size(0)
                correct_dis += (predicted_dis == Zs).sum().item()

            print('Accuracy of the Classifier: %d %%' % (100 * correct_cla / total_cla))
            print('Accuracy of the Discriminator: %d %%' % (100 * correct_dis / total_dis))

torch.save({
    'Encoder': encoder.state_dict(),
    'Decoder': decoder.state_dict(),
    'Classifier': classifier.state_dict(),
    'Discriminator': discriminator.state_dict(),
    'optimizer_E': optimizer_E.state_dict(),
    'optimizer_Dec': optimizer_Dec.state_dict(),
    'optimizer_C': optimizer_C.state_dict(),
    'optimizer_Dis': optimizer_Dis.state_dict(),
}, './saved_models/compas.pt')
