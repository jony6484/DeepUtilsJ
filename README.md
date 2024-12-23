# DeepUtils
Collected deep learning utils

by Jonathan Fuchs

# Training:
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