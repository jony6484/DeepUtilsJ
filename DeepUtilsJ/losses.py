import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDLoss():
    def __init__(self, return_sum=False):
        self.return_sum = return_sum

    def __call__(self, mu, log_var):
        kld = mu**2 + log_var.exp() - 1 - log_var
        if not self.return_sum:
            return 0.5 * kld.mean()
        return 0.5 * kld.sum()
        

class VAELoss():
    def __init__(self, beta, recon_loss=nn.MSELoss(), return_kld_sum=False):
        """
        recon_loss is the reconstruction loss: usualy MSE / CE, defualt reduction is mean,
        return_kld_sum is whether kld returns sum or mean, should match the recon_loss reduction
        """
        self.beta = beta
        self.recon_loss = recon_loss
        self.is_sum = return_kld_sum
        self.kld_loss = KLDLoss(return_sum=return_kld_sum)

    def __call__(self, X_hat, mu, log_var, X):
        MSE = self.recon_loss(X_hat, X)
        KLD = self.kld_loss(mu, log_var)
        loss = (1 - self.beta) * MSE + self.beta * KLD
        if self.is_sum:
            loss /= len(X)
        return loss


class CVAELoss():
    def __init__(self, beta, gamma, recon_loss=nn.MSELoss(), return_kld_sum=False):
        """
        recon_loss is the reconstruction loss: usualy MSE / CE, defualt reduction is mean,
        return_kld_sum is whether kld returns sum or mean, should match the recon_loss reduction
        """
        self.VAE_loss = VAELoss(beta, recon_loss, return_kld_sum)
        self.CE_loss = nn.CrossEntropyLoss()
        self.gamma = gamma

    def __call__(self, X_hat, mu, logVar, X, Y_hat, Y):
        VAE = self.VAE_loss(X_hat, mu, logVar, X)
        CE  = self.CE_loss(Y_hat, Y)
        return (1 - self.gamma) * VAE + self.gamma * CE