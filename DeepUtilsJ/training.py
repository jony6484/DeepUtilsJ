import numpy as np
import torch
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from .utils import validate_dir
from torchinfo import summary


class RiskOverwriteException(Exception):
    def __init__(self, message):            
        super().__init__(message)


class Trainer():
    def __init__(self, model, loss_func, metric_func, optimizer, scheduler=None, 
                 model_dir="./",
                 model_name="model",
                 loader_output_names=['X', 'Y'], 
                 model_input_names=['X'], 
                 model_output_names=['X_hat'],
                 loss_input_names=['Y_hat', 'Y'], 
                 metric_input_names=['Y_hat', 'Y'],
                 epoch_output_names = None,
                 plot_output_names = None,
                 save_all_output_plots = False,
                 plot_training_curve = False
                 ):
        self.model = model
        self.model_name = model_name
        self.loss_func = loss_func
        self.metric_func = metric_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader_output_names = loader_output_names
        self.model_input_names = model_input_names
        self.model_output_names = model_output_names
        self.loss_input_names = loss_input_names
        self.metric_input_names = metric_input_names
        self.epoch_output_names = epoch_output_names
        self.plot_output_names = plot_output_names
        self.save_all_output_plots = save_all_output_plots
        self.plot_training_curve = plot_training_curve
        self.figs = {}
        self.model_dir = validate_dir(model_dir)
        self.checkpoint_path = self.model_dir / "checkpoint.pt"
        self.last_epoch_path = self.model_dir / "last_epoch.pt"
        self.plots_dir = validate_dir(self.model_dir / 'plots')
        self.checkpoint_flag = None
        self.checkpoints = []
    
    def print_model_summary(self, loader):
        indx = np.random.randint(len(loader.dataset))
        dummy_shape = loader.dataset[indx][0].shape
        model_sum = summary(self.model, dummy_shape, batch_dim=0)
        with (self.model_dir / "model_summary.txt").open('a', encoding="utf-8") as file:
            file.write(str(model_sum))

    def train(self, train_loader, valid_loader, n_epochs=2, try_resume=True):
        # print model summary:
        self.print_model_summary(train_loader)
        # Dealing with resume:
        is_empty = not self.last_epoch_path.is_file()
        if is_empty:
            start_epoch = 0
            train_loss = np.full(n_epochs, np.nan)
            train_metric = np.full(n_epochs, np.nan)
            valid_loss = np.full(n_epochs, np.nan)
            valid_metric = np.full(n_epochs, np.nan)
            best_metric = -float('inf')
        elif try_resume:
            try: 
                # load last epoch
                start_epoch, best_metric, training_curves = self.load_checkpoint(path=self.last_epoch_path)
                start_epoch += 1
                train_loss = np.pad(training_curves['train_loss'], (0, n_epochs), mode='constant', constant_values=np.nan)
                train_metric = np.pad(training_curves['train_metric'], (0, n_epochs), mode='constant', constant_values=np.nan)
                valid_loss = np.pad(training_curves['valid_loss'], (0, n_epochs), mode='constant', constant_values=np.nan)
                valid_metric = np.pad(training_curves['valid_metric'], (0, n_epochs), mode='constant', constant_values=np.nan)
            except:
                print('could not load last epoch')
                return
        else:
            raise RiskOverwriteException("direcory not empty and 'try_resume=False'")
        if self.plot_training_curve:
            self.figs['loss'] = go.Figure()
            self.figs['metric'] = go.Figure()
        if self.plot_output_names is not None:
            for output_name in self.plot_output_names:
                self.figs[output_name] = make_subplots(rows=1, cols=2, subplot_titles=(f"Train - {output_name}", f"Valid - {output_name}"))
        # Epochs:
        self.epoch_counter = 0
        for epoch_i in range(start_epoch, start_epoch + n_epochs):
            self.epoch_counter = epoch_i
            start_time = time.time()
            # Train
            train_epoch_outputs = self.epoch(train_loader, train_mode=True)
            train_loss[epoch_i] = train_epoch_outputs['loss']
            train_metric[epoch_i] = train_epoch_outputs['metric']
            # Valid
            valid_epoch_outputs = self.epoch(valid_loader, train_mode=False)
            valid_loss[epoch_i] = valid_epoch_outputs['loss']
            valid_metric[epoch_i]  = valid_epoch_outputs['metric']
            # Time
            epoch_time = time.time() - start_time
            # Checkpoint
            training_curves = dict(train_loss=train_loss[:epoch_i+1], valid_loss=valid_loss[:epoch_i+1], train_metric=train_metric[:epoch_i+1], valid_metric=valid_metric[:epoch_i+1]) 
            if valid_metric[epoch_i] > best_metric:
                self.checkpoint_flag = True
                self.checkpoints.append(epoch_i)
                best_metric = valid_metric[epoch_i]
                self.save_checkpoint(epoch_i=epoch_i, best_metric=best_metric, training_curves=training_curves, path=self.checkpoint_path)
                if self.plot_output_names is not None:
                    self.plot_outputs(train_epoch_outputs, valid_epoch_outputs)
            else:
                self.checkpoint_flag = False
            # Print  
            self.epoch_print(epoch_i, start_epoch + n_epochs, epoch_time, train_loss[epoch_i], train_metric[epoch_i], valid_loss[epoch_i], valid_metric[epoch_i])
            if self.plot_training_curve:
                self.plot_training(train_loss, train_metric, valid_loss, valid_metric)
            # Save last epoch everythime for future training
            self.save_checkpoint(epoch_i=epoch_i, best_metric=best_metric, training_curves=training_curves, path=self.last_epoch_path)
        # Reloading and returning the model  
        _, _, training_curves = self.load_checkpoint(path=self.checkpoint_path)
        return self.model
    
    def save_checkpoint(self, epoch_i, best_metric, training_curves, path):
        checkpoint = {
        'epoch': epoch_i,
        'best_metric': best_metric,
        'model_state': self.model.state_dict(),
        'optimizer_state': self.optimizer.state_dict(),
        'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
        'rng_states': {
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate()
            },
        'training_curves': training_curves,
        'checkpoints': self.checkpoints
        }
        try:    torch.save(checkpoint, path)
        except: print("error: can't save checkpoint")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        torch.set_rng_state(checkpoint['rng_states']['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['rng_states']['cuda_rng'])
        np.random.set_state(checkpoint['rng_states']['numpy_rng'])
        random.setstate(checkpoint['rng_states']['python_rng'])
        epoch_i = checkpoint['epoch']
        best_metric = checkpoint['best_metric']
        training_curves = checkpoint['training_curves']
        self.checkpoints = checkpoint['checkpoints']
        return epoch_i, best_metric, training_curves

    def outputs_dict_converter(self, outputs, output_names, device=None):
        if not (isinstance(outputs, list) or isinstance(outputs, tuple)):
            outputs = [outputs]
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
        epoch_outputs = {}
        if self.epoch_output_names is not None:
            epoch_outputs.update({key: [] for key in self.epoch_output_names})
        for ii, loader_outputs in enumerate(data_loader):
            loader_outputs = self.outputs_dict_converter(loader_outputs, self.loader_output_names, device=device)
            if train_mode:
                model_outputs = self.model(*map(loader_outputs.get, self.model_input_names))
                model_outputs = self.outputs_dict_converter(model_outputs, self.model_output_names)
                combined_outputs = loader_outputs | model_outputs
                loss = self.loss_func(*map(combined_outputs.get, self.loss_input_names))
                if self.epoch_output_names is not None:
                    for key in self.epoch_output_names:
                        epoch_outputs[key].append(combined_outputs[key].detach().cpu())
                
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
                    if self.epoch_output_names is not None:
                        for key in self.epoch_output_names:
                            epoch_outputs[key].append(combined_outputs[key].detach().cpu())
            with torch.no_grad():
                N_batch = list(loader_outputs.values())[1].shape[0]
                count += N_batch
                epoch_loss += N_batch * loss.item()
                epoch_metric += N_batch * self.metric_func(*map(combined_outputs.get, self.metric_input_names))
            print(f'\r{"Train" if train_mode else "Val"} - Iteration: {ii:3d} ({nIter}): loss = {loss:2.6f}', end='')
        print('', end='\r')
        epoch_loss /= count
        epoch_metric /= count
        epoch_outputs['loss'] = epoch_loss
        epoch_outputs['metric'] = epoch_metric
        if self.epoch_output_names is not None:
            for key in self.epoch_output_names:
                epoch_outputs[key] = torch.cat(epoch_outputs[key])
        return epoch_outputs
    
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
        if self.checkpoint_flag:
            print(' <-- Checkpoint!')
        else:
            print("")

    def plot_training(self, train_loss, train_metric, valid_loss, valid_metric):
        train_outputs = {'loss': train_loss, 'metric': train_metric}
        valid_outputs = {'loss': valid_loss, 'metric': valid_metric}
        for output_name in ['loss', 'metric']:
            self.figs[output_name].data = []
            self.figs[output_name].update_layout(title=f"{output_name} visualization at Epoch {self.epoch_counter}",
                                                    plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"),
                                                    legend=dict(bordercolor='white', borderwidth=1))
            path_to_file = self.plots_dir.absolute() / f"{self.model_name}__{output_name}__visualization.html"
            for subset, outputs in zip(["Train", "Valid"], [train_outputs, valid_outputs]):
                self.figs[output_name].add_trace(go.Scatter(y=outputs[output_name], 
                                                                mode='lines',
                                                                name=f'{subset}'))
            # Only once for valid metrics
            if output_name == 'metric':
                self.figs[output_name].add_trace(go.Scatter(x=self.checkpoints, y=outputs[output_name][self.checkpoints], 
                                                                mode='markers',
                                                                marker=dict(symbol='x', size=8, color='yellow'),
                                                                name=f'chkp'))
            self.figs[output_name].write_html(path_to_file, auto_open=False)

    def plot_outputs(self, train_epoch_outputs, valid_epoch_outputs):
        with torch.no_grad():
            for output_name in self.plot_output_names:
                self.figs[output_name].data = []
                self.figs[output_name].update_layout(title=f"{output_name} visualization at Epoch {self.epoch_counter}",
                                                     plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"),
                                                     legend=dict(bordercolor='white', borderwidth=1))
                path_to_file = self.plots_dir.absolute() / f"{self.model_path.stem}__{output_name}__visualization.html"
                for subset, col, epoch_outputs in zip(["Train", "Valid"], [1, 2], [train_epoch_outputs, valid_epoch_outputs]):
                    marker_dict = dict(colorscale='Viridis', size=3, showscale=False)
                    if 'Y' in epoch_outputs:
                        marker_dict['color'] = epoch_outputs['Y']
                    self.figs[output_name].add_trace(go.Scatter(x=epoch_outputs[output_name][:, 0].detach().cpu().numpy(), 
                                                                y=epoch_outputs[output_name][:, 1].detach().cpu().numpy(), 
                                                                mode='markers', marker=marker_dict,
                                                                name=f'{subset} - Epoch {self.epoch_counter}'), row=1, col=col)
                self.figs[output_name].write_html(path_to_file, auto_open=False)
                self.figs[output_name].update_xaxes(showgrid=False, showline=False, zeroline=False)
                self.figs[output_name].update_yaxes(showgrid=False, showline=False, zeroline=False)
                if self.save_all_output_plots:
                    path_to_file_ii = self.plots_dir.absolute() / f"{self.epoch_counter}__{self.model_path.stem}__{output_name}__visualization.html"
                    self.figs[output_name].write_html(path_to_file_ii, auto_open=False)

                  
                  
                  


        
