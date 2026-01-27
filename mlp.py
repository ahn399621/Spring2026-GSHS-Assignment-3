import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

def set_seed(seed=42):
    '''
    Sets the random seed for reproducibility.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_input_tensors() -> tuple:
    '''
    Converts numpy arrays to PyTorch tensors.
    Returns: (x_tensor, y_tensor)
    '''
    # XOR inputs
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # XOR outputs: 0,1,1,0  (shape: (4,1))
    y = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=np.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return x_tensor, y_tensor

def implement_xor() -> nn.Module:
    '''
    Implements a simple XOR neural network using MLP.

    Returns:
        Trained MLP model on XOR dataset
    '''
    set_seed(42)

    # get the input and output tensors
    X, Y = get_input_tensors()

    # Define a 2 layer MLP model for XOR
    input_dim = 2
    hidden_dim = 4   # 2도 가능하지만 4가 더 안정적으로 학습됨
    output_dim = 1

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.Sigmoid()
    )

    # set other training parameters
    criterion = nn.BCELoss()
    epochs = 2000

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Train the model
    for _ in range(epochs):
        optimizer.zero_grad()

        # Make predictions
        predictions = model(X)

        # calculate the loss
        loss = criterion(predictions, Y)

        # Backpropagation
        loss.backward()

        # Update the weights
        optimizer.step()

    return model
