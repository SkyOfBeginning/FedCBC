import numpy as np
import torch
from torch import nn
import models.loss_functions as lf

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super().__init__()
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size)

    def forward(self, x):
        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def generate(self, z):

        recon_x = self.decoder(z)

        return recon_x

    def estimate_loglikelihood_single(self, x, S=5000, batch_size=32):
        x = x.to("cuda")
        with torch.no_grad():
            z_mu, z_logvar = self.encoder(x)
        repeats = int(np.ceil(S / batch_size))
        for rep in range(repeats):
            batch_size_current = (S % batch_size) if rep == (repeats - 1) else batch_size

            z = self.reparameterize(z_mu.expand(batch_size_current, -1), z_logvar.expand(batch_size_current, -1))
            # Calculate log_p_z
            with torch.no_grad():
                log_p_z = lf.log_Normal_standard(z, average=False, dim=1)

            log_q_z_x = lf.log_Normal_diag(z, mean=z_mu, log_var=z_logvar, average=False, dim=1)
            with torch.no_grad():
                x_recon = self.decoder(z)
            log_p_x_z = lf.log_Normal_standard(x=x, mean=x_recon, average=False, dim=-1)
            log_likelihoods = log_p_x_z * log_p_z / log_q_z_x
            all_lls = torch.cat([all_lls, log_likelihoods]) if rep > 0 else log_likelihoods

        log_likelihood = all_lls.logsumexp(dim=0) - np.log(S)

        return log_likelihood


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.Tanh())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()

        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.Tanh())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):


        x = self.MLP(z)

        return x