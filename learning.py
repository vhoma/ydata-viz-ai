import nn_architecture
import data_loader as dl
from data_loader import LOG_LEVELS, set_log_level

import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import json
from tempfile import gettempdir

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from clearml import Task
#Task.set_offline(offline_mode=True)
task = Task.init(project_name="viz", task_name="test local toggle batch norm")
clearml_logger = task.get_logger()


def get_device():
    if torch.cuda.is_available():
        device_name = "cuda:0"
    # elif torch.backends.mps.is_available():
    #     device_name = "mps"
    else:
        device_name = "cpu"
    return torch.device(device_name)


def weighted_mse_loss(result, target, weight):
    return ((weight * (result - target)) ** 2).sum()


class Learner:
    def __init__(self, data_path, batch_size, batch_size_val, num_epochs, learning_rate, scheduler_input,
                 min_val, max_val, model_state_file, transform_angle_schedule,
                 best_loss_threshold, nonrandom_val_step, batchnorm_on, loss_weights):
        self.data_path = data_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        print(transform_angle_schedule)  # debug
        self.transform_angle_schedule = json.loads(transform_angle_schedule)
        self.best_loss_threshold = best_loss_threshold
        self.nonrandom_val_step = nonrandom_val_step
        self.loss_weights = torch.Tensor(json.loads(loss_weights))

        # connect to GPU
        self.device = get_device()
        logging.info(f"device: {self.device}")

        # get data loader for train and val
        data_path_train = os.path.join(data_path, "train")
        data_path_val = os.path.join(data_path, "val")
        default_angle_limit = max(self.transform_angle_schedule.values())
        self.dataset_train = dl.Img3dDataSet(data_path_train, min_val, max_val, self.device,
                                             max_transform_angle=default_angle_limit)
        self.dataset_val = dl.Img3dDataSet(data_path_val, min_val, max_val, self.device,
                                           max_transform_angle=default_angle_limit)
        self.data_loader = {
            'train': DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True),
            'val': DataLoader(self.dataset_val, batch_size=batch_size_val, shuffle=False)
        }

        # model init
        #model = nn_architecture.SiamAirNet()
        model = nn_architecture.Siam_AirNet2(batchnorm_on=batchnorm_on)

        # load model state if needed
        if model_state_file:
            model.load_state_dict(torch.load(model_state_file, map_location=torch.device('cpu')))

        self.model = model.to(self.device)
        self.loss_weights = self.loss_weights.to(self.device)

        # other training vars
        #self.criterion = nn.MSELoss(reduction='sum')
        self.criterion = weighted_mse_loss

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        #self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.scheduler = eval(scheduler_input)

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

    def train_step(self, phase, x, y, matrix, target_angle):
        # Zero the gradients
        self.optimizer.zero_grad()

        # load to the device
        # x = x.to(self.device)
        # y = y.to(self.device)
        # matrix = matrix.to(self.device)

        # run the model
        res = self.model(x, y)

        # calculate batch loss
        loss = self.criterion(res, target_angle, self.loss_weights)

        if phase == 'train':
            loss.backward()
            self.optimizer.step()

        # log
        batch_loss = loss.item() / x.shape[0]  # self.batch_size
        self.epoch_loss_list.append(batch_loss)
        logging.debug(f"Epoch #{self.current_epoch}, phase: {phase}, batch #{self.batch_num}: Current loss {batch_loss}\n")
        if phase == "train":
            # this logging costs too much...
            # clearml_logger.report_scalar(
            #     title="batch_loss",
            #     series=f"batch_LOSS",
            #     value=batch_loss,
            #     iteration=self.batch_num + self.current_epoch * len(self.data_loader[phase])
            # )
            self.loss_history.append(loss.item())
        self.batch_num += 1

    def train_epoch(self, phase):
        if phase == "train":
            # save learning rate
            logging.info(f"Learning rate: {self.scheduler.get_last_lr()}")
            clearml_logger.report_scalar(
                title="learning_rate",
                series=f"learning_rate",
                value=self.scheduler.get_last_lr()[0],
                iteration=self.current_epoch
            )

            # set new transform angle limit if needed
            scheduled_transform_angle_limit = self.transform_angle_schedule.get(str(self.current_epoch))
            if scheduled_transform_angle_limit:
                self.dataset_train.max_transform_angle = scheduled_transform_angle_limit
            clearml_logger.report_scalar(
                title="transform_angle_limit",
                series=f"angle_limit",
                value=self.dataset_train.max_transform_angle,
                iteration=self.current_epoch
            )

        self.batch_num = 0
        self.epoch_loss_list = []
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        # run through all data
        if phase == "val" and self.is_fixed_validation_epoch():
            # once in a while will run an epoch with this flag on
            self.dataset_val.use_fixed_angles = True
        for x, y, matrix, angle in self.data_loader[phase]:   # iterate through data loader here
            angle = angle.to(self.device)
            self.train_step(phase, x, y, matrix, angle)
        self.dataset_val.use_fixed_angles = False   # always switch off

        if phase == "train":
            self.scheduler.step()

        # log epoch loss
        epoch_loss = np.array(self.epoch_loss_list).mean()
        logging.info(f"Epoch #{self.current_epoch}, phase: {phase}, Epoch loss {epoch_loss}\n")
        if phase == "val" and self.is_fixed_validation_epoch():
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                if self.best_loss < self.best_loss_threshold:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(gettempdir(), f"best_model_{self.current_epoch}_{self.best_loss:.2f}.pt")
                    )
            clearml_logger.report_scalar(
                title="loss",
                series=f"val_fixed_LOSS",
                value=epoch_loss,
                iteration=self.current_epoch
            )
        else:
            clearml_logger.report_scalar(
                title="loss",
                series=f"{phase}_epoch_LOSS",
                value=epoch_loss,
                iteration=self.current_epoch
            )

    def train(self):
        self.reset_vars()  # in case this is not the first time

        try:
            # Iterate throughout the epochs
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                for phase in ['train', 'val']:
                    self.train_epoch(phase)
        finally:
            torch.save(
                self.model.state_dict(),
                os.path.join(gettempdir(), f"best_model_{self.current_epoch}_final.pt")
            )

    def is_fixed_validation_epoch(self):
        return self.current_epoch % self.nonrandom_val_step == 0


def main(data):
    # set logging level
    set_log_level(data['log_level'])

    # set random seed
    np.random.seed(data.get('seed', None))

    # parse boolean args
    batchnorm_on = not("false" in data['batchnorm_on'].lower())

    # train
    learner = Learner(
        data_path=data['data_path'],
        num_epochs=data['num_epochs'],
        learning_rate=data['learning_rate'],
        min_val=data['min_val'],
        max_val=data['max_val'],
        batch_size=data['batch_size'],
        batch_size_val=data['batch_size_val'],
        model_state_file=data['model_state_file'],
        transform_angle_schedule=data['transform_angle_schedule'],
        scheduler_input=data['scheduler_input'],
        best_loss_threshold=data['best_loss_threshold'],
        nonrandom_val_step=data['nonrandom_val_step'],
        batchnorm_on=batchnorm_on,
        loss_weights=data['loss_weights']
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
    parser.add_argument('--num-epochs', required=False, type=int, default=1, help='Number of epochs')
    parser.add_argument('--min-val', required=False, type=int, default=-1000, help='Min value for normalization')
    parser.add_argument('--max-val', required=False, type=int, default=1000, help='Max value for normalization')
    parser.add_argument('--seed', required=False, type=int, help='Random seed')
    parser.add_argument('--log-level', default="info", choices=LOG_LEVELS.keys(), help='Logging level, default "info"')
    parser.add_argument('--model-state-file', required=False, default=None, help='Path to .pt file with model weights.')
    parser.add_argument(
        '--transform-angle-schedule',
        required=False,
        default="{\"0\"': 45}",
        help='JSON dict where key is epoch and value is transform angle limit change on that epoch.'
    )
    parser.add_argument(
        '--scheduler-input',
        required=False,
        default="lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)",
        help='Code to create LR schedule. SORRY MOM, i know this is very insecure'
    ),
    parser.add_argument(
        '--best-loss-threshold',
        required=False,
        default=20,
        help='Model state will be saved only if validation loss is lower then this threshold'
    ),
    parser.add_argument(
        '--nonrandom-val-step',
        required=False,
        default=50,
        help='Once in <step> epochs we will run validation phase on pre generated transform angles, for consistent validation.'
    ),
    parser.add_argument(
        '--batchnorm-on',
        required=False,
        default="true",
        help='Toggle batch normalization on regression layers.'
    ),
    parser.add_argument(
        '--loss-weights',
        required=False,
        default="[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]",
        help='Weights for weighted MSE loss.'
    )
    return vars(parser.parse_args())


if '__main__' == __name__:
    args = get_args()
    print(args)
    main(args)
