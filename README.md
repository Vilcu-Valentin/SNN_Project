# Spiking Neural Networks with snnTorch & Norse

This repo contains two simple Spiking Neural Network (SNN) implementations on the MNIST dataset:

- `snntorch` version: trains an SNN using leaky integrate-and-fire (LIF) neurons over 25 time steps.
- `norse` version: loads a trained model and classifies test images using Poisson spike encoding over 150 steps.
