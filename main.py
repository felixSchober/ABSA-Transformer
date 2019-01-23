
import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg'
matplotlib.rcParams['backend.qt5'] = 'PyQt5'

import logging
import torch

from data.data_loader import Dataset
from data.germeval2017 import germeval2017_dataset

from misc.preferences import PREFERENCES
from misc.visualizer import *
from misc.run_configuration import get_default_params
from misc import utils

from optimizer import get_default_optimizer
from criterion import NllLoss

from models.transformer.encoder import TransformerEncoder
from models.softmax_output import SoftmaxOutputLayer, OutputLayer, SoftmaxOutputLayerWithCommentWiseClass
from models.transformer_tagger import TransformerTagger
from models.transformer.train import Trainer

experiment_name = 'just-testing'
print('\n\nABSA Transformer\n\n')

PREFERENCES.defaults(
    data_root='./data/germeval2017',
    data_train='train_v1.4.tsv',    
    data_validation='dev_v1.4.tsv',
    data_test='test_TIMESTAMP1.tsv',
    early_stopping='highest_5_F1'
)

hyperparameters = get_default_params()
hyperparameters.model_size = 300
hyperparameters.batch_size = 12
hyperparameters.early_stopping = -1
hyperparameters.use_cuda = True
hyperparameters.language = 'de'
hyperparameters.num_epochs = 25
hyperparameters.log_every_xth_iteration = -1


experiment_name = utils.create_loggers(experiment_name=experiment_name)

dataset = Dataset(
    'germeval',
    logging.getLogger('dataset'),
    hyperparameters,
    source_index=0,
    target_vocab_index=1,
    data_path=PREFERENCES.data_root,
    train_file=PREFERENCES.data_train,
    valid_file=PREFERENCES.data_validation,
    test_file=PREFERENCES.data_test,
    file_format='.tsv',
    init_token=None,
    eos_token=None
)
dataset.load_data(germeval2017_dataset)


loss = NllLoss(dataset.target_size, weight=dataset.class_weights)
transformer = TransformerEncoder(dataset.source_embedding,
                                 hyperparameters=hyperparameters)
tagging_softmax = SoftmaxOutputLayerWithCommentWiseClass(hyperparameters.model_size, dataset.target_size)
model = TransformerTagger(transformer, tagging_softmax)

optimizer = get_default_optimizer(model, hyperparameters)
trainer = Trainer(
                    model,
                    loss,
                    optimizer,
                    hyperparameters,
                    dataset,
                    experiment_name,
                    enable_tensorboard=True)

result = trainer.train(hyperparameters.num_epochs, use_cuda=hyperparameters.use_cuda, perform_evaluation=False)
evaluation_results = trainer.perform_final_evaluation()
