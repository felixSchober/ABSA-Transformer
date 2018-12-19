from data.conll import conll2003_dataset, extract_samples
from misc.preferences import PREFERENCES
from misc.visualizer import *
from misc.hyperparameters import get_default_params
from optimizer import get_default_optimizer
from misc import utils
from models.transformer.encoder import TransformerEncoder
from models.softmax_output import SoftmaxOutputLayer, OutputLayer
from models.transformer_tagger import TransformerTagger
from models.transformer.train import Trainer
from criterion import NllLoss

import torch
experiment_name = 'just-testing'
PREFERENCES.defaults(
    data_root='./data/conll2003',
    data_train='eng.train.txt',
    data_validation='eng.testa.txt',
    data_test='eng.testb.txt',
    early_stopping='highest_5_F1'
)

hyper_parameters = get_default_params()
hyper_parameters.model_size = 300
hyper_parameters.batch_size = 80
hyper_parameters.early_stopping = -1
experiment_name = utils.create_loggers(experiment_name=experiment_name)

conll2003 = conll2003_dataset('ner', hyper_parameters.batch_size,
                              root=PREFERENCES.data_root,
                              train_file=PREFERENCES.data_train,
                              validation_file=PREFERENCES.data_validation,
                              test_file=PREFERENCES.data_test,
                              use_cuda=False)
samples = extract_samples(conll2003['examples'])



# 10 words with a 100-length embedding
target_vocab = conll2003['vocabs'][0]
target_size = len(target_vocab)

loss = NllLoss(target_size)
# transformer = GoogleTransformer(True, target_size, target_size, num_units, 2, 2, 512, 0.1)
transformer = TransformerEncoder(conll2003['embeddings'][0],
                                 n_enc_blocks=2,
                                 n_head=3,
                                 d_model=hyper_parameters.model_size,
                                 d_k=100,
                                 d_v=100)
tagging_softmax = SoftmaxOutputLayer(hyper_parameters.model_size, target_size)
model = TransformerTagger(transformer, tagging_softmax)

#test_sample_iter = iterate_with_sample_data(conll2003['iters'][0], 50)
#df = predict_some_examples_to_df(model, test_sample_iter, num_samples=50)


optimizer = get_default_optimizer(model, hyper_parameters)
trainer = Trainer(model,
                    loss,
                    optimizer,
                    hyper_parameters,
                    conll2003['iters'],
                    experiment_name,
                    log_every_xth_iteration=-1,
                    enable_tensorboard=True,
                    dummy_input=conll2003['dummy_input'])
trainer.train(1, True)
