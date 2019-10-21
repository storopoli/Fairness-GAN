#!/usr/bin/env python
import argparse
import numpy as np
import pickle

from torch.utils.data import Dataset

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

# Load on GPU / CPU
if cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Model Saved PATH
saved_models = torch.load(f"./saved_models/compas-alpha_{opt.alpha_value}-beta_{opt.beta_value}-gamma_{opt.gamma_value}.pt", map_location=device)


class DatasetCompas(Dataset):
    def __init__(self, ds):
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[:, 0:-1]  # all expect last column Y
        Z = self.data[:, -2]  # only penultimate column Z
        Y = self.data[:, -1]  # only last column Y
        sample = {'X': X, 'Z': Z, 'Y': Y}
        return sample


train_dataset = DatasetCompas(ds)


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

# Initialize networks and load models parameters
encoder = Encoder()
encoder.load_state_dict(saved_models['Encoder'])

decoder = Decoder()
decoder.load_state_dict(saved_models['Decoder'])

classifier = Classifier()
classifier.load_state_dict(saved_models['Classifier'])

discriminator = Discriminator()
discriminator.load_state_dict(saved_models['Discriminator'])


# CUDA
if cuda:
    encoder.cuda()
    decoder.cuda()
    classifier.cuda()
    discriminator.cuda()
    BCE_loss.cuda()
    MSE_loss.cuda()
    CrossEntropy_loss.cuda()

# Optimizers loaded from saved models
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_E.load_state_dict(saved_models['optimizer_E'])

optimizer_Dec = torch.optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Dec.load_state_dict(saved_models['optimizer_Dec'])

optimizer_C = torch.optim.Adam(
    classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_C.load_state_dict(saved_models['optimizer_C'])

optimizer_Dis = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Dis.load_state_dict(saved_models['optimizer_Dis'])


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.

encoder.eval()
decoder.eval()
classifier.eval()
discriminator.eval()

correct_cla = 0
total_cla = 0
correct_dis = 0
total_dis = 0

x = Tensor(train_dataset['data']['X'])
z = Tensor(train_dataset['data']['Z'])
y = Tensor(train_dataset['data']['Y'])

# adding an extra dimension to Z and Y (M, 1)
z = z.unsqueeze_(-1)
y = y.unsqueeze_(-1)

if cuda:
    x = x.cuda()
    z = z.cuda()
    y = y.cuda()

# Generate x_tilde
x_tilde = encoder(x)

# Classify x_tilde for Y
y_hat = classifier(x_tilde)

# Discriminate x_tilde for Z
z_hat = discriminator(x_tilde)


# Create a list of lists to Plot
accuracy_cla = []
accuracy_dis = []
for t in range(0, 100):
    threshold = t / 100
    # threshold = Variable(Tensor([thresh]))

    # Accuracy Classifier
    matches_cla = [(i > threshold) == j for i, j in zip(y_hat.float(), y.float())]
    acc_cla = (threshold, matches_cla.count(True) / len(matches_cla))
    accuracy_cla.append(acc_cla)

    # Accuracy Classifier
    matches_dis = [(i > threshold) == j for i, j in zip(z_hat.float(), z.float())]
    acc_dis = (threshold, matches_dis.count(True) / len(matches_dis))
    accuracy_dis.append(acc_dis)

# Save lists
with open('accuracy_cla.pickle', 'wb') as handle:
    pickle.dump(accuracy_cla, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('accuracy_dis.pickle', 'wb') as handle:
    pickle.dump(accuracy_dis, handle, protocol=pickle.HIGHEST_PROTOCOL)
