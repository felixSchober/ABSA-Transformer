
import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg'
matplotlib.rcParams['backend.qt5'] = 'PyQt5'

from data.conll import conll2003_dataset, extract_samples
from data.germeval2017 import germeval2017_dataset
from misc.preferences import PREFERENCES
from misc.visualizer import *
from misc.hyperparameters import get_default_params
from optimizer import get_default_optimizer
from misc import utils
from models.transformer.encoder import TransformerEncoder
from models.softmax_output import SoftmaxOutputLayer, OutputLayer, SoftmaxOutputLayerWithCommentWiseClass
from models.transformer_tagger import TransformerTagger
from models.transformer.train import Trainer
from criterion import NllLoss

import torch
experiment_name = 'just-testing'
PREFERENCES.defaults(
    data_root='./data/germeval2017',
    data_train='train_v1.4.tsv',    
    data_validation='dev_v1.4.tsv',
    data_test='test_TIMESTAMP1.tsv',
    early_stopping='highest_5_F1'
)
use_cuda = True
hyper_parameters = get_default_params()
hyper_parameters.model_size = 300
hyper_parameters.batch_size = 5
hyper_parameters.early_stopping = -1
experiment_name = utils.create_loggers(experiment_name=experiment_name)

germeval2017_dataset = germeval2017_dataset( hyper_parameters.batch_size,
                              root=PREFERENCES.data_root,
                              train_file=PREFERENCES.data_train,
                              validation_file=PREFERENCES.data_validation,
                              test_file=PREFERENCES.data_test,
                              use_cuda=use_cuda)
#samples = extract_samples(conll2003['examples'])



# 10 words with a 100-length embedding
target_vocab = germeval2017_dataset['vocabs'][1]
target_size = len(target_vocab)

loss = NllLoss(target_size)
# transformer = GoogleTransformer(True, target_size, target_size, num_units, 2, 2, 512, 0.1)
transformer = TransformerEncoder(germeval2017_dataset['embeddings'][0],
                                 n_enc_blocks=2,
                                 n_head=3,
                                 d_model=hyper_parameters.model_size,
                                 d_k=100,
                                 d_v=100)
tagging_softmax = SoftmaxOutputLayerWithCommentWiseClass(hyper_parameters.model_size, target_size)
model = TransformerTagger(transformer, tagging_softmax)

# test_sample_iter = iterate_with_sample_data(conll2003['iters'][1], 200)
# test_sample_iter = iterate_with_sample_data(conll2003['iters'][1], 200)

# df = predict_some_examples_to_df(model, conll2003['iters'][1], num_samples=50)


optimizer = get_default_optimizer(model, hyper_parameters)
trainer = Trainer(
                    target_size,
                    model,
                    loss,
                    optimizer,
                    hyper_parameters,
                    germeval2017_dataset['iters'],
                    experiment_name,
                    log_every_xth_iteration=-1,
                    enable_tensorboard=False,
                    dummy_input=germeval2017_dataset['dummy_input'])
result = trainer.train(2, use_cuda=use_cuda, perform_evaluation=False)
evaluation_results = trainer.perform_final_evaluation()
