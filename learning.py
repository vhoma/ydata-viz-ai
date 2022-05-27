import nn_architecture
import data_loader as dl
from data_loader import LOG_LEVELS, set_log_level

import logging
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse
import sys
from scipy.ndimage import affine_transform

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from clearml import Task
task = Task.init(project_name="viz", task_name="test run")
clearml_logger = task.get_logger()


def get_device():
    if torch.cuda.is_available():
        device_name = "cuda:0"
    # elif torch.backends.mps.is_available():
    #     device_name = "mps"
    else:
        device_name = "cpu"
    return torch.device(device_name)


class Learner:
    def __init__(self, data_path, batch_size, num_epochs, learning_rate, step_size, gamma, min_val, max_val):
        self.data_path = data_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # connect to GPU
        self.device = get_device()
        logging.info(f"device: {self.device}")

        # get data loader
        dataset = dl.Img3dDataSet(data_path, min_val, max_val)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # model init
        model = nn_architecture.SiamAirNet()
        self.model = model.to(self.device)

        # other training vars
        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        # get input for test run
        #self.x, self.y, self.matrix = next(iter(self.data_loader))

    def train(self):
        best_loss = np.inf
        loss_history = []

        # Iterate throughout the epochs
        for epoch in range(self.num_epochs):
            batch_num = 0
            for x, y, matrix in self.data_loader:
                # Zero the gradients
                self.optimizer.zero_grad()

                # load to the device
                x = x.to(self.device)
                y = y.to(self.device)
                matrix = matrix.to(self.device)

                # run the model
                res = self.model(x, y)

                # calculate batch loss
                loss = self.criterion(res, matrix.flatten(start_dim=1))
                loss.backward()

                # Optimize
                self.optimizer.step()
                self.scheduler.step()
                batch_loss = loss.item() / self.batch_size
                logging.info(f"Epoch #{epoch}, batch #{batch_num}: Current loss {batch_loss}\n")
                clearml_logger.report_scalar(
                    title="loss",
                    series=f"Epoch#{epoch}",
                    value=batch_loss,
                    iteration=batch_num
                )
                batch_num += 1
                loss_history.append(loss.item())

        return loss_history


#def main(data_path, learning_rate, step_size, gamma, num_epochs, min_val, max_val, log_level, seed=None):
def main(data):
    # set logging level
    set_log_level(data['log_level'])

    # set random seed
    np.random.seed(data.get('seed', None))

    # train
    learner = Learner(
        data_path=data['data_path'],
        num_epochs=data['num_epochs'],
        learning_rate=data['learning_rate'],
        step_size=data['step_size'],
        gamma=data['gamma'],
        min_val=data['min_val'],
        max_val=data['max_val'],
        batch_size=data['batch_size']
    )

    logging.info("Start training!")
    loss_history = learner.train()
    logging.info("Finished training!")

    # plot loss
    plt.plot(loss_history)
    plt.show()

    # # check
    # trained_matrix = learner.model(learner.x.to(learner.device), learner.y.to(learner.device))
    # trained_matrix = trained_matrix.cpu().detach().numpy()
    # trained_matrix = np.append(trained_matrix, [0, 0, 0, 1]).reshape(4, 4)
    # logging.info(f"original matrix: \n{learner.matrix.numpy()}")
    # logging.info(f"trained matrix: \n{trained_matrix}")
    #
    # x_nmp = learner.x.detach().numpy()[0].transpose(1, 2, 0)
    # y_nmp = learner.y.detach().numpy()[0].transpose(1, 2, 0)
    # plt.imshow(y_nmp[:, :, 10])
    # plt.show()
    #
    # x_new = affine_transform(x_nmp, trained_matrix)
    # plt.imshow(x_new[:, :, 10])
    # plt.show()


def get_args():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--data-path', required=False, type=str, default="./data100", help='Working directory')
    parser.add_argument('--batch_size', required=False, type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', required=False, type=float, default=0.005, help='Min value for normalization')
    parser.add_argument('--step-size', required=False, type=int, default=10, help='Scheduler step size')
    parser.add_argument('--gamma', required=False, type=float, default=0.8, help='Gamma value for scheduler')
    parser.add_argument('--num-epochs', required=False, type=int, default=1, help='Number of epochs')
    parser.add_argument('--min-val', required=False, type=int, default=-1000, help='Min value for normalization')
    parser.add_argument('--max-val', required=False, type=int, default=1000, help='Max value for normalization')
    parser.add_argument('--seed', required=False, type=int, help='Random seed')
    parser.add_argument('--log-level', default="info", choices=LOG_LEVELS.keys(), help='Logging level, default "info"')
    return vars(parser.parse_args())


if '__main__' == __name__:
    args = get_args()
    print(args)
    main(args)
