import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class LinearVae(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers=[20, 20], decoder_layers=[20, 20]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Encoder:
        layer_lst = []
        for output_dim in encoder_layers:
            layer_lst += [nn.Linear(input_dim, output_dim), nn.ReLU()]
            input_dim = output_dim
        layer_lst.append(nn.Linear(input_dim, latent_dim * 2))
        self.encoder = nn.Sequential(*layer_lst)
        # Decoder:
        input_dim = latent_dim
        layer_lst = []
        for output_dim in decoder_layers:
            layer_lst += [nn.Linear(input_dim, output_dim), nn.ReLU()]
            input_dim = output_dim
        layer_lst.append(nn.Linear(input_dim, self.input_dim))       
        self.decoder = nn.Sequential(*layer_lst)

    def forward(self, X):
        X = self.encoder(X)
        mu, log_var = X.chunk(2, dim=1)
        if self.training == True:
            e = torch.randn_like(mu)
            sigma = torch.exp(.5 * log_var)
            Z = sigma * e + mu
        else:
            Z = mu
        X_hat = self.decoder(Z)
        return X_hat, mu, log_var


class AEDataset(Dataset):
    def __init__(self, X):
        self.X = X
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X[idx]


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


def r2_score(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
  
    return 1 - (ss_residual / ss_total) if ss_total != 0 else 0


class VaeReducer:
    def __init__(self, input_dim, latent_dim, encoder_layers=[20, 20], decoder_layers=[20, 20], valid_size=0.2):
        self.reducer = LinearVae(input_dim, latent_dim, encoder_layers, decoder_layers)
        self.valid_size = valid_size
        self.to_cpu()

    def to_gpu(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reducer.to(self.device)

    def to_cpu(self):
        self.device = "cpu"
        self.reducer.to(self.device)

    def get_reducer_device(self):
        return str(next(self.reducer.parameters()).device)

    def fix_input_type_and_device(self, X):
        if isinstance(X, np.ndarray):
            X_type = np.ndarray
            X = torch.tensor(X, dtype=torch.float)
        else:
            X_type = torch.Tensor
        X_device = str(X.device)
        X = X.to(self.device)
        return X, X_type, X_device

    def revert_input_type_and_device(self, X, X_type, X_device):
        X = X.detach()
        if X_type == np.ndarray:
            X = X.cpu().numpy()
        elif X_device != X.device:
            X.to(X_device)
        return X
            
    def fit(self, X, max_batch_size=None, beta=0.5, max_epochs=100, target_r2=0.5):
        X, X_type, X_device = self.fix_input_type_and_device(X)
        X = X.to(self.device)
        inds = torch.randperm(len(X))
        N_valid = int(len(X) * self.valid_size)
        data_train = AEDataset(X[inds[N_valid:]])
        data_valid = AEDataset(X[inds[:N_valid]])
        if max_batch_size is None:
            batch_size = len(data_train)
        loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        loader_valid = DataLoader(data_valid, batch_size=batch_size, shuffle=False)
        loss_func = VAELoss(beta=beta)
        metric_func = r2_score
        optimizer = torch.optim.Adam(self.reducer.parameters())
        epoch_metric = 0
        epoch_count = 0
        pbar = tqdm(range(max_epochs), desc="Processing")
        for ii in pbar:
            pbar.set_description(f"r2: {epoch_metric:0.4f}")
            if epoch_metric >= target_r2:
                break         
            epoch_count += 1
            # Train
            self.reducer.train(True)
            for X, X in loader_train:
                X_hat, mu, log_var = self.reducer(X)
                loss = loss_func(X_hat, mu, log_var, X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Valid
            self.reducer.train(False)
            epoch_metric = 0
            count = 0
            for X, X in loader_valid:
                with torch.no_grad():
                    X_hat, mu, log_var = self.reducer(X)
                    metric = metric_func(X, X_hat)
                    N_batch = X.shape[0]
                    count += N_batch
                    epoch_metric += N_batch * metric
            epoch_metric /= count
        X = self.revert_input_type_and_device(X, X_type, X_device)
        print(f"data fitted with r2 of {epoch_metric}, after {epoch_count} epochs")


    def transform(self, X):
        X, X_type, X_device = self.fix_input_type_and_device(X)
        with torch.no_grad():
            X = self.reducer.encoder(X)
            mu, log_var = X.chunk(2, dim=1)
        X = self.revert_input_type_and_device(X, X_type, X_device)
        mu= self.revert_input_type_and_device(mu, X_type, X_device)
        return mu
