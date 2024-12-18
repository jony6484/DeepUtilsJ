import numpy as np
import torch
import time
from pathlib import Path
import pandas as pd


def epoch_print(epoch_i, n_epochs, epoch_time, train_loss, train_metric, val_loss, val_metric):
        if epoch_i % 10 == 0:
            print('-' * 130)
        print('Epoch '            f'{epoch_i       :03d}:',  end='')
        print(' | Train loss: '   f'{train_loss   :6.3f}',   end='')
        print(' | Val loss: '     f'{val_loss     :6.3f}',   end='')
        print(' | Train Metric: ' f'{train_metric :6.3f}',   end='')
        print(' | Val Metric: '   f'{val_metric   :6.3f}',   end='')
        print(' | epoch time: '   f'{epoch_time   :6.3f}',   end='')
        print(' | time left: '   f'{epoch_time * (n_epochs - epoch_i)   :6.3f} |', end='')


def epoch(model, data_loader, loss_func, metric_func, optimizer=None, scheduler=None, train_mode=False):
    model.train(train_mode)
    epoch_loss = 0
    epoch_metric = 0
    count = 0
    nIter = len(data_loader)
    device = next(model.parameters()).device
    for ii, (X, Y) in enumerate(data_loader):
        X = X.to(device)
        Y = Y.to(device)
        
        if train_mode:
            Y_hat = model(X)
            loss = loss_func(Y_hat, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                Y_hat = model(X)
                loss = loss_func(Y_hat, Y)
        with torch.no_grad():
            N_batch = X.shape[0]
            count += N_batch
            epoch_loss += N_batch * loss.item()
            epoch_metric += N_batch * metric_func(Y_hat, Y)
        print(f'\r{"Train" if train_mode else "Val"} - Iteration: {ii:3d} ({nIter}): loss = {loss:2.6f}', end='')
    print('', end='\r')
    epoch_loss /= count
    epoch_metric /= count
    return epoch_loss, epoch_metric


def VAE_epoch(model, data_loader, loss_func, metric_func, optimizer=None, scheduler=None, train_mode=True):
    model.train(train_mode)
    epoch_loss = 0
    epoch_metric = 0
    count = 0
    nIter = len(data_loader)
    device = next(model.parameters()).device
    #-- Iterate over the mini-batches:
    for ii, (X, _) in enumerate(data_loader):
        #-- Move to device (CPU\GPU):
        X = X.to(device)

        if train_mode:
            #-- Forward:
            X_hat, mu, log_var = model(X)
            loss               = loss_func(X_hat, mu, log_var, X)

            #-- Backward:
            optimizer.zero_grad() #-- set gradients to zeros
            loss     .backward () #-- backward
            optimizer.step     () #-- update parameters
            if scheduler is not None:
                scheduler.step     () #-- update learning rate

        else:
            with torch.no_grad():
                #-- Forward:
                X_hat, mu, log_var = model(X)
                loss               = loss_func(X_hat, mu, log_var, X)

        with torch.no_grad():
            N_batch       = X.shape[0]
            count        += N_batch
            epoch_loss   += N_batch * loss.item()
            epoch_metric += N_batch * metric_func(X_hat, X)
        print(f'\r{"Train" if train_mode else "Val"} - Iteration: {ii:3d} ({nIter}): loss = {loss:2.6f}', end='')
    print('', end='\r')
    epoch_loss   /= count
    epoch_metric /= count
    return epoch_loss, epoch_metric


def CVAE_epoch(model, data_loader, loss_func, metric_func, optimizer=None, scheduler=None, train_mode=True):
    model.train(train_mode)
    epoch_loss = 0
    epoch_metric = 0
    count = 0
    nIter = len(data_loader)
    device = next(model.parameters()).device
    #-- Iterate over the mini-batches:
    for ii, (X, Y) in enumerate(data_loader):
        #-- Move to device (CPU\GPU):
        X = X.to(device)
        Y = Y.to(device)
        if train_mode == True:
            #-- Forward:
            X_hat, mu, log_var, Y_hat = model(X)
            loss               = loss_func(X_hat, mu, log_var, X, Y_hat, Y)

            #-- Backward:
            optimizer.zero_grad() #-- set gradients to zeros
            loss     .backward () #-- backward
            optimizer.step     () #-- update parameters
            if scheduler is not None:
                scheduler.step     () #-- update learning rate

        else:
            with torch.no_grad():
                #-- Forward:
                X_hat, mu, log_var, Y_hat = model(X)
                loss               = loss_func(X_hat, mu, log_var, X, Y_hat, Y)

        with torch.no_grad():
            N_batch       = X.shape[0]
            count        += N_batch
            epoch_loss   += N_batch * loss.item()
            epoch_metric += N_batch * metric_func(X_hat, X)
        print(f'\r{"Train" if train_mode else "Val"} - Iteration: {ii:3d} ({nIter}): loss = {loss:2.6f}', end='')
    print('', end='\r')
    epoch_loss   /= count
    epoch_metric /= count
    return epoch_loss, epoch_metric


def train_model(model, loss_func, metric_func, optimizer, train_loader, valid_loader, scheduler=None, n_epochs=2, model_path=".\model.pt", epoch=epoch):
    model_path = Path(model_path)
    train_loss = np.full(n_epochs, np.nan)
    train_metric = np.full(n_epochs, np.nan)
    valid_loss = np.full(n_epochs, np.nan)
    valid_metric = np.full(n_epochs, np.nan)
    best_metric = -float('inf') 
    # Epochs:
    for epoch_i in range(n_epochs):
        start_time = time.time()
        # Train
        train_epoch_loss, train_epoch_metric = epoch(model, train_loader, loss_func, metric_func, optimizer, scheduler, train_mode=True)
        train_loss[epoch_i] = train_epoch_loss
        train_metric[epoch_i] = train_epoch_metric
        # Valid
        valid_epoch_loss, valid_epoch_metric = epoch(model, valid_loader, loss_func, metric_func, optimizer, scheduler, train_mode=False)
        valid_loss[epoch_i] = valid_epoch_loss
        valid_metric[epoch_i]  = valid_epoch_metric
        # Time
        epoch_time = time.time() - start_time
        # Print & Save
        epoch_print(epoch_i, n_epochs, epoch_time, train_epoch_loss, train_epoch_metric, valid_epoch_loss, valid_epoch_metric)
        if valid_epoch_metric > best_metric:
            best_metric = valid_epoch_metric  
            try   : torch.save(model.state_dict(), model_path)
            except: pass
            print(' <-- Checkpoint!')
        else:
            print('')   
    # Reloading and returning the model  
    model.load_state_dict(torch.load(model_path))
    return model, pd.DataFrame(dict(train_loss=train_loss, valid_loss=valid_loss, train_metric=train_metric))






        




