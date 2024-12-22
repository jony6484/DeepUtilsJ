import numpy as np
import torch
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class Trainer():
    def __init__(self, model, loss_func, metric_func, optimizer, scheduler=None, 
                 loader_output_names=['X', 'Y'], 
                 model_input_names=['X'], 
                 model_output_names=['X_hat'],
                 loss_input_names=['Y_hat', 'Y'], 
                 metric_input_names=['Y_hat', 'Y']):
        self.model = model
        self.loss_func = loss_func
        self.metric_func = metric_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader_output_names = loader_output_names
        self.model_input_names = model_input_names
        self.model_output_names = model_output_names
        self.loss_input_names = loss_input_names
        self.metric_input_names = metric_input_names
    
    def train(self, train_loader, valid_loader, n_epochs=2, model_path=".\model.pt"):
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
            epoch_outputs = self.epoch(train_loader, train_mode=True)
            train_loss[epoch_i] = epoch_outputs['loss']
            train_metric[epoch_i] = epoch_outputs['metric']
            # Valid
            epoch_outputs = self.epoch(valid_loader, train_mode=False)
            valid_loss[epoch_i] = epoch_outputs['loss']
            valid_metric[epoch_i]  = epoch_outputs['metric']
            # Time
            epoch_time = time.time() - start_time
            # Print & Save
            self.epoch_print(epoch_i, n_epochs, epoch_time, train_loss[epoch_i], train_metric[epoch_i], valid_loss[epoch_i], valid_metric[epoch_i])
            if valid_metric[epoch_i] > best_metric:
                best_metric = valid_metric[epoch_i]  
                try   : torch.save(self.model.state_dict(), model_path)
                except: pass
                print(' <-- Checkpoint!')
            else:
                print('')   
        # Reloading and returning the model  
        self.model.load_state_dict(torch.load(model_path))
        return self.model, pd.DataFrame(dict(train_loss=train_loss, valid_loss=valid_loss, train_metric=train_metric))
    
    def outputs_dict_converter(self, outputs, output_names, device=None):
        output_dict = {key: output for key, output in zip(output_names, outputs)}
        if device is not None:
            for key in output_dict.keys():
                output_dict[key] = output_dict[key].to(device)
        return output_dict
    
    def epoch(self, data_loader, train_mode=False):
        self.model.train(train_mode)
        epoch_loss = 0
        epoch_metric = 0
        count = 0
        nIter = len(data_loader)
        device = next(self.model.parameters()).device
        for ii, loader_outputs in enumerate(data_loader):
            loader_outputs = self.outputs_dict_converter(loader_outputs, self.loader_output_names, device=device)
            if train_mode:
                model_outputs = self.model(*map(loader_outputs.get, self.model_input_names))
                model_outputs = self.outputs_dict_converter(model_outputs, self.model_output_names)
                combined_outputs = loader_outputs | model_outputs
                loss = self.loss_func(*map(combined_outputs.get, self.loss_input_names))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            else:
                with torch.no_grad():
                    model_outputs = self.model(*map(loader_outputs.get, self.model_input_names))
                    model_outputs = self.outputs_dict_converter(model_outputs, self.model_output_names)
                    combined_outputs = loader_outputs | model_outputs
                    loss = self.loss_func(*map(combined_outputs.get, self.loss_input_names))
            with torch.no_grad():
                N_batch = list(loader_outputs.values())[0].shape[0]
                count += N_batch
                epoch_loss += N_batch * loss.item()
                epoch_metric += N_batch * self.metric_func(*map(combined_outputs.get, self.metric_input_names))
            print(f'\r{"Train" if train_mode else "Val"} - Iteration: {ii:3d} ({nIter}): loss = {loss:2.6f}', end='')
        print('', end='\r')
        epoch_loss /= count
        epoch_metric /= count
        return dict(loss=epoch_loss, metric=epoch_metric)
    
    def epoch_print(self, epoch_i, n_epochs, epoch_time, train_loss, train_metric, val_loss, val_metric):
        if epoch_i % 10 == 0:
            print('-' * 130)
        print('Epoch '            f'{epoch_i       :03d}:',  end='')
        print(' | Train loss: '   f'{train_loss   :6.3f}',   end='')
        print(' | Val loss: '     f'{val_loss     :6.3f}',   end='')
        print(' | Train Metric: ' f'{train_metric :6.3f}',   end='')
        print(' | Val Metric: '   f'{val_metric   :6.3f}',   end='')
        print(' | epoch time: '   f'{epoch_time   :6.3f}',   end='')
        print(' | time left: '   f'{epoch_time * (n_epochs - epoch_i)   :6.3f} |', end='')

        
        
