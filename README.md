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
# Bulid
python setup.py bdist_wheel