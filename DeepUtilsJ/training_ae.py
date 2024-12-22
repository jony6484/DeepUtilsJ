import numpy as np
import torch
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def VAE_epoch(model, data_loader, loss_func, metric_func, optimizer=None, scheduler=None, train_mode=True):
    model.train(train_mode)
    epoch_loss = 0
    epoch_metric = 0
    count = 0
    nIter = len(data_loader)
    device = next(model.parameters()).device
    # collect embeddings and labels for print
    labels = []
    embeddings = []
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
        # For scatter plot the embeddings:
        labels.append(Y.detach().cpu())
        embeddings.append(mu.detach().cpu())
        with torch.no_grad():
            N_batch       = X.shape[0]
            count        += N_batch
            epoch_loss   += N_batch * loss.item()
            epoch_metric += N_batch * metric_func(X_hat, X)
        print(f'\r{"Train" if train_mode else "Val"} - Iteration: {ii:3d} ({nIter}): loss = {loss:2.6f}', end='')
    print('', end='\r')
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    epoch_loss   /= count
    epoch_metric /= count
    return dict(loss=epoch_loss, metric=epoch_metric, embeddings=embeddings, labels=labels)


def CVAE_epoch(model, data_loader, loss_func, metric_func, optimizer=None, scheduler=None, train_mode=True):
    model.train(train_mode)
    epoch_loss = 0
    epoch_metric = 0
    count = 0
    nIter = len(data_loader)
    device = next(model.parameters()).device
    # collect embeddings and labels for print
    labels = []
    embeddings = []
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
        # For scatter plot the embeddings:
        labels.append(Y.detach().cpu())
        embeddings.append(mu.detach().cpu())
        with torch.no_grad():
            N_batch       = X.shape[0]
            count        += N_batch
            epoch_loss   += N_batch * loss.item()
            epoch_metric += N_batch * metric_func(X_hat, X)
        print(f'\r{"Train" if train_mode else "Val"} - Iteration: {ii:3d} ({nIter}): loss = {loss:2.6f}', end='')
    print('', end='\r')
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    epoch_loss   /= count
    epoch_metric /= count
    return dict(loss=epoch_loss, metric=epoch_metric, embeddings=embeddings, labels=labels)


def train_model(model, loss_func, metric_func, optimizer, train_loader, valid_loader, scheduler=None, n_epochs=2, model_path=".\model.pt", epoch=epoch, plot_embeddings=False):
    model_path = Path(model_path)
    train_loss = np.full(n_epochs, np.nan)
    train_metric = np.full(n_epochs, np.nan)
    valid_loss = np.full(n_epochs, np.nan)
    valid_metric = np.full(n_epochs, np.nan)
    best_metric = -float('inf') 
    if plot_embeddings:
        plt.ion()
        fig_train, ax_train = plt.subplots(figsize=(8, 6))
        fig_valid, ax_valid = plt.subplots(figsize=(8, 6))
    # Epochs:
    for epoch_i in range(n_epochs):
        start_time = time.time()
        # Train
        train_outs = epoch(model, train_loader, loss_func, metric_func, optimizer, scheduler, train_mode=True)
        train_loss[epoch_i] = train_outs['loss']
        train_metric[epoch_i] = train_outs['metric']
        # Valid
        valid_outs = epoch(model, valid_loader, loss_func, metric_func, train_mode=False)
        valid_loss[epoch_i] = valid_outs['loss']
        valid_metric[epoch_i]  = valid_outs['metric']
        # Plots
        if plot_embeddings:
            train_embs = train_outs['embeddings']
            train_labels = train_outs['labels']
            valid_embs = valid_outs['embeddings']
            valid_labels = valid_outs['labels']
            if train_embs.shape[1] > 2:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2).fit(train_embs)
                train_embs = reducer.transform(train_embs)
                valid_embs = reducer.transform(valid_embs)
            embeddings_plot(train_embs, train_labels, ax_train)
            embeddings_plot(valid_embs, valid_labels, ax_valid)
        # Time
        epoch_time = time.time() - start_time
        # Print & Save
        epoch_print(epoch_i, n_epochs, epoch_time, train_loss[epoch_i], train_metric[epoch_i], valid_loss[epoch_i], valid_metric[epoch_i])
        if valid_metric[epoch_i] > best_metric:
            best_metric = valid_metric[epoch_i]  
            try   : torch.save(model.state_dict(), model_path)
            except: pass
            print(' <-- Checkpoint!')
        else:
            print('')   
    # Reloading and returning the model  
    model.load_state_dict(torch.load(model_path))
    if plot_embeddings:
        plt.ioff()
        plt.show()
    return model, pd.DataFrame(dict(train_loss=train_loss, valid_loss=valid_loss, train_metric=train_metric))


def embeddings_plot(embeddings, labels, ax):
    ax.clear()
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.pause(0.1)