import nn_architecture
import data_loader as dl

import logging
import numpy as np
import copy

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


def train(model, x, y, matrix):
    best_loss = np.inf
    loss_history = []

    # Iterate throughout the epochs
    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        matrix_gpu = matrix.to(device)

        # Pass in the two images into the network and obtain two outputs
        res = model(x_gpu, y_gpu)

        # Pass the outputs of the networks and label into the loss function
        loss = criterion(res, matrix_gpu.flatten())
        # Calculate the backpropagation
        loss.backward()

        # Optimize
        optimizer.step()
        scheduler.step()
        logging.info(f"Epoch number {epoch}\n Current loss {loss.item()}\n")

        if best_loss > loss:
            best_loss = loss

        loss_history.append(loss.item())
    logging.info(f'Best loss {best_loss}')
    return loss_history


if __name__ == "__main__":
    # set logging level
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    # set vars
    home_path = "./data"

    # connect to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    # get data loader
    dataset = dl.Img3dDataSet(home_path, -1000, 1000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # set training vars
    model = nn_architecture.SiamAirNet()
    model = model.to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    num_epochs = 100

    # train
    x, y, matrix = next(iter(dataloader))
    train(model, x, y, matrix)

    # check
    trained_matrix = model(x.to(device), y.to(device))
    trained_matrix = trained_matrix.cpu().detach().numpy()
    trained_matrix = np.append(trained_matrix, [0, 0, 0, 1]).reshape(4, 4)
    logging.info(f"original matrix: \n{matrix.numpy()}")
    logging.info(f"trained matrix: \n{trained_matrix}")
