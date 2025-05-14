# building-micrograd

This is a minimal implementation of a multi-layer perceptron (MLP). It was built following a lesson by Andrej Kaparty about the basics of machine learning.

## Overview

- `Value`: Scalar value object that tracks data and gradients through operations.
- `Neuron`, `Layer`, `MLP`: Components that build up a basic neural network.
- Includes a simple training loop with manual input for number of epochs and learning rate.
- Demonstrates forward and backward passes with live loss updates.

## Requirements

No external libraries are required.

## How to Run

>>> python interface.py

You will be prompted to enter number of epochs and a learning rate
