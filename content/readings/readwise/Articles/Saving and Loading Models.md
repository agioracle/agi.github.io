---
author: Matthew Inkawhich
title: Saving and Loading Models
category: #articles
url: https://pytorch.org/tutorials/beginner/saving_loading_models.html
---
![rw-book-cover](https://rdl.ink/render/https%3A%2F%2Fpytorch.org%2Ftutorials%2Fbeginner%2Fsaving_loading_models.html)

## Metadata
- Author: Matthew Inkawhich
- Full Title: Saving and Loading Models
- Category: #articles
- Summary: To save and load models in PyTorch, you should initialize the model and optimizer, then use torch.save() to save the model's state_dict and optimizer's state_dict to a file. When loading, initialize the models and optimizers again and use torch.load() to load the saved dictionary. Remember to set the model to evaluation mode with model.eval() before running inference and to training mode with model.train() if you wish to resume training. Additionally, you can warmstart model training using parameters from a different model to help convergence.
- URL: https://pytorch.org/tutorials/beginner/saving_loading_models.html

## Highlights
- A *state_dict* is simply a Python dictionary object that maps each layer to its parameter tensor. Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) and registered buffers (batchnorm’s running_mean) have entries in the model’s *state_dict*. Optimizer objects (`torch.optim`) also have a *state_dict*, which contains information about the optimizer’s state, as well as the hyperparameters used. ([View Highlight](https://read.readwise.io/read/01j1pgc0g03y05jbj24v99sg2s))
- When saving a model for inference, it is only necessary to save the trained model’s learned parameters. Saving the model’s *state_dict* with the `torch.save()` function will give you the most flexibility for restoring the model later, which is why it is the recommended method for saving models. ([View Highlight](https://read.readwise.io/read/01j1pgv5eaek2dy36r95cah0yt))
- pickle does not save the model class itself. Rather, it saves a path to the file containing the class, which is used during load time. Because of this, your code can break in various ways when used in other projects or after refactors. ([View Highlight](https://read.readwise.io/read/01j1phdantq8s0ns8ckz30p5s0))
- One common way to do inference with a trained model is to use [TorchScript](https://pytorch.org/docs/stable/jit.html), an intermediate representation of a PyTorch model that can be run in Python as well as in a high performance environment like C++. TorchScript is actually the recommended model format for scaled inference and deployment. ([View Highlight](https://read.readwise.io/read/01j1phf1yxsffvd150t5ap6rqs))
- Remember that you must call `model.eval()` to set dropout and batch normalization layers to evaluation mode before running inference. ([View Highlight](https://read.readwise.io/read/01j1phrabd2818ckfay7k4nvkz))
