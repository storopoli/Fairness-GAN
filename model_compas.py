#!/usr/bin/env python
import argparse
import numpy as np
import time

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn
import torch

from load_compas_data import *
from plot_metrics import *

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
parser.add_argument("--n_cpu", type=int, default=4,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--alpha_value", type=float, default=1,
                    help="alpha regularizer for classification utility")
parser.add_argument("--beta_value", type=float, default=1,
                    help="beta regularizer for decoder reconstruction of the inputs")
parser.add_argument("--gamma_value", type=float, default=1,
                    help="gamma regularizer for fair classification")

opt = parser.parse_args()
print('\n', opt)

SEED = 1234
seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)  # for reproducibility

# CUDA Stuff
cuda = True if torch.cuda.is_available() else False

if cuda:
    print("\nRunning on the GPU\n")
else:
    print("\nRunning on the CPU\n")

X, Y, Z = load_compas_data()

ds = np.c_[X, Z['race'], Y]

# Y is "two_year_recid"
# X are ['age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25', 'sex', 'priors_count', 'c_charge_degree'] all 0 or 1 and priors count is continuous (mean=0; variance=1)
# Z is 'race' 0 White; 1 Black


class DatasetCompas(Dataset):
    def __init__(self, ds):
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index, 0:-2]  # all expect last columns Y and Z
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
            nn.Linear(7, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 7),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 7),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(7, 8),
            nn.LeakyReLU(0.2, inplace=True),
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
            nn.Linear(7, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.model(x)
        return z


# Loss functions
BCE_loss = torch.nn.BCELoss()
MSE_loss = torch.nn.MSELoss()

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

# Logging
MODEL_NAME = f"compas-alpha_{opt.alpha_value}-beta_{opt.beta_value}-gamma_{opt.gamma_value}"

with open(f"./logs/{MODEL_NAME}.csv", "w") as f:
        f.write(f"{MODEL_NAME},timestamp,epoch,enc_loss,dec_loss,cla_loss,cla_acc,dis_loss,dis_acc\n")

# Training
for epoch in range(opt.n_epochs):
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        x = batch['X'].float()
        z = batch['Z'].float()
        y = batch['Y'].float()

        # adding an extra dimension to Z and Y (M, 1)
        z = z.view(-1, 1)
        y = y.view(-1, 1)

        # Concatenating X and Z for Encoder
        x = torch.cat((x, z), -1)
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

    # Accuracy Classifier
    X_hat = encoder(Tensor(np.c_[X, Z['race']])).detach()
    Y_hat = classifier(X_hat).detach()
    matches_cla = [(i > 0.5) == j for i, j in zip(Y_hat, Tensor(Y))]
    acc_cla = matches_cla.count(True) / len(matches_cla)

    # Accuracy Discriminator
    Z_hat = discriminator(X_hat).detach()
    matches_dis = [(i > 0.5) == j for i, j in zip(Z_hat, Tensor(Z['race']))]
    acc_dis = matches_dis.count(True) / len(matches_dis)

    print(
        "[Epoch %d/%d] [enc loss: %f] [dec loss: %f] [cla loss: %f] [dis loss: %f] [cla acc: %f] [dis acc: %f] [Time: %f]"
        % (epoch, opt.n_epochs, enc_loss.item(), dec_loss.item(), cla_loss.item(), dis_loss.item(), acc_cla, acc_dis, time_taken)
    )

    with open(f"./logs/{MODEL_NAME}.csv", "a") as f:
        f.write(f"{MODEL_NAME},{int(time.time())},{epoch},{round(float(enc_loss),4)},{round(float(dec_loss),4)},{round(float(cla_loss),4)},{round(float(acc_cla),2)},{round(float(dis_loss),4)},{round(float(acc_dis),2)}\n")

# Plot the graph after the end of training
create_acc_loss_graph(f"{MODEL_NAME}")

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
