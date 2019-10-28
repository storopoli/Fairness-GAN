#!/usr/bin/env python
import argparse
import numpy as np

from torch.utils.data import Dataset

import torch.nn as nn
import torch


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
parser.add_argument("--alpha_value", type=float, default=1,
                    help="alpha regularizer for classification utility")
parser.add_argument("--beta_value", type=float, default=1,
                    help="beta regularizer for decoder reconstruction of the inputs")
parser.add_argument("--gamma_value", type=float, default=1,
                    help="gamma regularizer for fair classification")


opt = parser.parse_args()
print(opt)

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)  # for reproducibility

# Load Data
npzfile = np.load('data/adult.npz')

X = np.concatenate([npzfile['x_train'], npzfile['x_test']])
print('X dimensions:', X.shape)

Z = np.concatenate([npzfile['attr_train'], npzfile['attr_test']])
print('Z dimensions:', Z.shape)

Y = np.concatenate([npzfile['y_train'], npzfile['y_test']])
print('Y dimensions:', Y.shape)

ds = np.c_[X, Z, Y]

# Y is 'income'
# X are 13 'age', 'workclass', 'education', 'education-num,marital-status', 'occupation', 'relationship','race', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
# Z is 'sex'

cuda = True if torch.cuda.is_available() else False

# Load on GPU / CPU
if cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Model Saved PATH
saved_models = torch.load(f"./saved_models/adult-alpha_{opt.alpha_value}-beta_{opt.beta_value}-gamma_{opt.gamma_value}.pt", map_location=device)


class DatasetAdult(Dataset):
    def __init__(self, ds):
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[:, 0:-2]  # all expect last two columns Y
        Z = self.data[:, -3]  # only third to last column Z
        Y = self.data[:, -2:]  # only last two columns Y
        sample = {'X': X, 'Z': Z, 'Y': Y}
        return sample


train_dataset = DatasetAdult(ds)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(113, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 113),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(114, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 113),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(113, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.model(x)
        return y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(113, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.model(x)
        return z


# Loss functions
BCE_loss = torch.nn.BCELoss()
MSE_loss = torch.nn.MSELoss()

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

x = Tensor(train_dataset['data']['X'])
z = Tensor(train_dataset['data']['Z'])
y = Tensor(train_dataset['data']['Y'])

# adding an extra dimension to Z (M, 1)
z = z.unsqueeze_(-1)

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

# Classify
matches_cla = [torch.argmax(i) == torch.argmax(j) for i, j in zip(y_hat, y)]
acc_cla = matches_cla.count(True) / len(matches_cla)

# Discriminator
matches_dis = [(i > 0.5) == j for i, j in zip(z_hat, z)]
acc_dis = matches_dis.count(True) / len(matches_dis)

print(f"Accuracy Classifier: {acc_cla}\nAccuracy Discriminator: {acc_dis}")
