
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
from criterion import NllLoss, LossCombiner

from models.transformer.encoder import TransformerEncoder
from models.softmax_output import SoftmaxOutputLayerWithCommentWiseClass
from models.transformer_tagger import TransformerTagger
from models.jointAspectTagger import JointAspectTagger
from models.transformer.train import Trainer

experiment_name = 'JointAspectTest'
print('\n\nABSA Transformer\n\n')
use_cuda = True
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
hyperparameters.num_encoder_blocks = 3
hyperparameters.n_heads = 6
hyperparameters.d_k = 50
hyperparameters.d_v = 50

hyperparameters.early_stopping = 5
hyperparameters.use_cuda = use_cuda
hyperparameters.language = 'de'
hyperparameters.num_epochs = 25
hyperparameters.log_every_xth_iteration = 500
hyperparameters.embedding_type = 'fasttext'

experiment_name = utils.create_loggers(experiment_name=experiment_name)

dataset = Dataset(
    'germeval',
    logging.getLogger('data_loaoder'),
    hyperparameters,
    source_index=0,
    target_vocab_index=2,
    data_path=PREFERENCES.data_root,
    train_file=PREFERENCES.data_train,
    valid_file=PREFERENCES.data_validation,
    test_file=PREFERENCES.data_test,
    file_format='.tsv',
    init_token=None,
    eos_token=None
)
dataset.load_data(germeval2017_dataset)

#loss = NllLoss(4, weight=dataset.class_weights)
loss = LossCombiner(4, dataset.class_weights, NllLoss)

transformer = TransformerEncoder(dataset.source_embedding,
								 hyperparameters=hyperparameters)
tagging_softmax = SoftmaxOutputLayerWithCommentWiseClass(hyperparameters.model_size, dataset.target_size)
model = JointAspectTagger(transformer, hyperparameters.model_size, 4, 20, dataset.target_names)
#model = TransformerTagger(transformer, tagging_softmax)

optimizer = get_default_optimizer(model, hyperparameters)
trainer = Trainer(
					model,
					loss,
					optimizer,
					hyperparameters,
					dataset,
					experiment_name,
					enable_tensorboard=True)

trainer.load_model()
trainer.set_cuda(True)
result_labels = trainer.classify_sentence('Die Bahn preise sind sehr billig')

#result = trainer.train(use_cuda=hyperparameters.use_cuda, perform_evaluation=False)
#trainer.perform_final_evaluation(False)

#evaluation_results = trainer.perform_final_evaluation()
print('Exit')