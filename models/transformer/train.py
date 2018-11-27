import os
import logging
from tensorboardX import SummaryWriter
import time
import os
import shutil
from typing import Tuple

from misc.utils import set_seeds, torch_summarize
from misc.hyperparameters import HyperParameters

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.autograd import *
from tqdm import tqdm

DEFAULT_CHECKPOINT_PATH = ''


class Trainer(object):

    def __init__(self,
                model: nn.Module, 
                loss: nn.Module,
                optimizer: torch.optim.Optimizer,
                parameters: HyperParameters,
                data_iterators: Tuple[torchtext.data.Iterator, torchtext.data.Iterator, torchtext.data.Iterator],
                early_stopping: int,
                experiment_name: str,
                seed: int = 42,
                enable_tensorboard: bool=True,
                dummy_input: torch.Tensor=None):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.parameters = parameters

        assert len(data_iterators) == 3
        self.train_iterator, self.valid_iterator, self.test_iterator = data_iterators
        self.experiment_name = experiment_name
        self.early_stopping = early_stopping

        self.checkpoint_dir = os.path.join(os.getcwd(), 'logs', experiment_name, 'checkpoints')


        self._reset()
        self.seed = seed
        self.enable_tensorboard = enable_tensorboard

        # this logger will not print to the console. Only to the file.
        self.logger = logging.getLogger(__name__)

        # this logger will both print to the console as well as the file
        self.logger_prediction = logging.getLogger('prediction')

        model_summary = torch_summarize(self.model)

        if enable_tensorboard:
            assert dummy_input is not None
            self.tb_writer = SummaryWriter(comment=self.experiment_name)

            # for now until add graph works (needs pytorch version >= 0.4) add the model description as text
            self.tb_writer.add_text('model', model_summary, 0)
            self.tb_writer.add_graph(self.model, dummy_input, verbose=True)

        # TODO: initialize the rest of the trainings parameters
        # https://github.com/kolloldas/torchnlp/blob/master/torchnlp/common/train.py

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_model_checkpoint = None
        self.early_stopping_counter = self.early_stopping
        self.loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self._reset_histories()

    def _step(self, input: torch.Tensor, target: torch.Tensor):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """

        # Clears the gradients of all optimized :class:`torch.Tensor` s
        self.optimizer.zero_grad()

        # Compute loss and gradient
        # TODO: Provide masks
        output = self.model(input, None)

        loss = self.loss(output, target)

        # preform training step
        loss.backward()
        self.optimizer.step()

        self.train_loss_history.append(loss.data[0])
        return loss

    def _check_accuracies(self) -> int:
        #TODO: 
        return 0

    def train(self, num_epochs: int, should_use_cuda: bool=False):
        # TODO: Support for early stopping
        set_seeds(self.seed)
        continue_training = True
        start_time = time.time()
        
        self.logger.info('START training.')

        for epoch in range(num_epochs):

            # Set up the batch generator for a new epoch
            self.train_iterator.init_epoch()
            self.epoch = epoch

            interation = 0
            # loop iterations
            for batch in tqdm(self.train_iterator, leave=False): # batch is torchtext.data.batch.Batch

                if not continue_training:
                    break

                interation += 1

                # Sets the module in training mode
                self.model.train()

                x, y = batch.inputs_word, batch.labels

                self._step(x, y)

            # at the end of each epoch, check the accuracies
            val_acc = self._check_accuracies()

            # early stopping if no improvement of val_acc during the last self.early_stopping epochs
            # https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

                # Save best model
                self.best_model_checkpoint = {
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                }
                self._save_checkpoint(interation)

                # restore early stopping counter
                self.early_stopping_counter = self.early_stopping

            else:
                self.early_stopping_counter -= 1

                # if early_stopping_counter is 0 restore best weights and stop training
                if self.early_stopping > -1 and self.early_stopping_counter <= 0:
                    print('> Early Stopping after {0} epochs of no improvements.'.format(self.early_stopping))
                    print('> Restoring params of best model with validation accuracy of: '
                            , self.best_val_acc)

                    # Restore best model
                    self.model.load_state_dict(self.best_model_checkpoint['state_dict'])
                    self.optimizer.load_state_dict(self.best_model_checkpoint['optimizer'])
                    continue_training = False
                    break


        self.logger.info('STOP training.')

        # At the end of training swap the best params into the model
        # Restore best model
        self.model.load_state_dict(self.best_model_checkpoint['state_dict'])
        self.optimizer.load_state_dict(self.best_model_checkpoint['optimizer'])

        if self.tb_writer is not None:
            self.tb_writer.export_scalars_to_json(os.path.join(os.getcwd(), 'logs', self.experiment_name, "model_all_scalars.json"))
            self.tb_writer.close()

    def _save_checkpoint(self, iteration: int) -> None:
        self.logger.debug('Saving model... ' + self.checkpoint_dir)

        checkpoint = {
            'iteration': iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        filename = 'checkpoint_{}.data'.format(iteration)
        try:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
            #shutil.copyfile(filename, os.path.join(self.checkpoint_dir, filename))
        except:
            self.logger.exception('Could not save model.')