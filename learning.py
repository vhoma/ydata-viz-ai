import nn_architecture
import data_loader as dl
from data_loader import LOG_LEVELS, set_log_level

import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from tempfile import gettempdir

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from clearml import Task
task = Task.init(project_name="viz", task_name="run_with_validation")
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
    def __init__(self, data_path, batch_size, batch_size_val, num_epochs, learning_rate, step_size, gamma, min_val, max_val):
        self.data_path = data_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # connect to GPU
        self.device = get_device()
        logging.info(f"device: {self.device}")

        # get data loader for train and val
        data_path_train = os.path.join(data_path, "train")
        data_path_val = os.path.join(data_path, "val")
        dataset_train = dl.Img3dDataSet(data_path_train, min_val, max_val)
        dataset_val = dl.Img3dDataSet(data_path_val, min_val, max_val)
        self.data_loader = {
            'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True),
            'val': DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False)
        }

        # model init
        #model = nn_architecture.SiamAirNet()
        model = nn_architecture.Siam_AirNet2()
        self.model = model.to(self.device)

        # other training vars
        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        # changing vars
        self.current_epoch = 0
        self.loss_history = []
        self.epoch_loss_list = []
        self.best_loss = np.inf
        self.batch_num = 0

    def reset_vars(self):
        self.current_epoch = 0
        self.loss_history = []
        self.epoch_loss_list = []
        self.best_loss = np.inf
        self.batch_num = 0

    def train_step(self, phase, x, y, matrix):
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

        if phase == 'train':
            loss.backward()
            self.optimizer.step()

        # log
        batch_loss = loss.item() / x.shape[0]  # self.batch_size
        self.epoch_loss_list.append(batch_loss)
        logging.info(f"Epoch #{self.current_epoch}, phase: {phase}, batch #{self.batch_num}: Current loss {batch_loss}\n")
        if phase == "train":
            clearml_logger.report_scalar(
                title="loss",
                series=f"batch_LOSS",
                value=batch_loss,
                iteration=self.batch_num + self.current_epoch * len(self.data_loader[phase])
            )
            self.loss_history.append(loss.item())
        self.batch_num += 1

    def train_epoch(self, phase):
        self.batch_num = 0
        self.epoch_loss_list = []
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        # run through all data
        for x, y, matrix in self.data_loader[phase]:
            self.train_step(phase, x, y, matrix)

        if phase == "train":
            self.scheduler.step()

        # log epoch loss
        epoch_loss = np.array(self.epoch_loss_list).mean()
        clearml_logger.report_scalar(
            title="loss",
            series=f"{phase}_epoch_LOSS",
            value=epoch_loss,
            iteration=self.current_epoch
        )
        if phase == "val" and epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            torch.save(
                self.model.state_dict(),
                os.path.join(gettempdir(), f"best_model_{self.current_epoch}_{self.best_loss:.2f}.pt")
            )

    def train(self):
        self.reset_vars()  # in case this is not the first time

        # Iterate throughout the epochs
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            for phase in ['train', 'val']:
                self.train_epoch(phase)


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
        batch_size=data['batch_size'],
        batch_size_val=data['batch_size_val']
    )

    logging.info("Start training!")
    learner.train()
    logging.info("Finished training!")

    # plot loss
    plt.plot(learner.loss_history)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--data-path', required=False, type=str, default="./data100", help='Working directory')
    parser.add_argument('--batch_size', required=False, type=int, default=4, help='Batch size')
    parser.add_argument('--batch_size_val', required=False, type=int, default=1, help='Batch size validation')
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
