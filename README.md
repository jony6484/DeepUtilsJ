# DeepUtils
Collected deep learning utils

by Jonathan Fuchs

# Training:
1. first create a Trainer instance,
2. train with the .train method
3. This trainer needs specific instructions for all inputs/outputs:
    please use basic names 'X', 'Y' when needed, all outputs/inputs must be tensors, names order must match the input/output order
    - loader_output_names: the output names for the data loder (which it gets from the Dataset object) ie. ['X', 'Y']
    - model_input_names: the name of the input(s) which the model requires, must match the loader output names, default ['X']
    - model_output_names: naming the model output, ie. ['Z'] or ['Y_hat'] or ['X_hat', 'Y_hat'] etc.
    - loss_input_names, metric_input_names: must match the corresponding model output names and loader output names ie. ['Y_hat', 'Y']
    - epoch_output_names: defaults to None, when required (for let's say plotting) epoch may return needed outputs.
    - plot_output_names: defaults to None, when specified, plots will be created (embeddigs) each checkpoint, names must be included      in the epoch output names, if 'Y' is present in the epoch outputs, embeddings will be colored per its classes.
    - plot_training_curve: defaults to True, if so - plots of the training curve will be saved each checkpoint (loss, metric)
    - model_path - a string for the relatinve/ absolute path. plots are saved under plots dir inside it.
```python
model_path = Path("models/CVAE.pt")
trainer = training.Trainer(model=model, loss_func=loss, metric_func=metric, optimizer=optimizer, model_path=model_path,
                               save_all_output_plots=False,
                               plot_training_curve=True,
                               loader_output_names=['X', 'Y'], 
                               model_input_names=['X'], 
                               model_output_names=['X_hat', 'mu', 'logVar', 'Y_hat'], 
                               loss_input_names=['X_hat', 'mu', 'logVar', 'X', 'Y_hat', 'Y'], 
                               metric_input_names=['X_hat', 'X'],
                               epoch_output_names=['mu', 'Y'],
                               plot_output_names=['mu'])
model, metrics = trainer.train(train_loader=loader_train, valid_loader=loader_valid, n_epochs=60)
```
If we wish to continue the training, all the data is saved inside the model.pt file, just run the trainer.train method with the "resume_train=True", the n_epochs are the additional repochs that will be added.
More Examples:
```python
reducer = VaeReducer(input_dim=10, latent_dim=2, target_r2=0.75)
trainer = Trainer(model=model, loss_func=loss_func, metric_func=metric_func, optimizer=optimizer,
                  model_dir=model_dir, model_name="windowCnn", 
                  loader_output_names=['X', 'Y'],
                  model_input_names=['X'], 
                  model_output_names=['Y_hat', 'Z'],
                  loss_input_names=['Y_hat', 'Y'],
                  metric_input_names=['Y_hat', 'Y'],
                  epoch_output_names=['Z', 'Y', 'Y_hat'], 
                  plot_output_names=['Z'], plot_training_curve=True, embedding_dim_reducer=reducer, max_plot_samples=5_000,
                  int2label_dict=data_train.int2label
                  )
model = trainer.train(loader_train, loader_valid, n_epochs=100)
```
Here we have some new args:
embedding_dim_reducer: if the output embedding dim is more than 2, default PCA is fitted on the Train data and applied on
the Train+Validation. it can be replaced with any dimentionality reduction model, which has fit and transform methods, and
could reduce from the embedding dim to 2 dimensions.

max_plot_samples: defaults to 3000, specifeis how many scatter points to plot in the outputs plot.

int2label_dict: if specified, is used to name the outputs via hoveron, must be a dict.

# Bulid
python setup.py bdist_wheel