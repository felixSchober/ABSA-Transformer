from data.conll import conll2003_dataset
from misc.preferences import PREFERENCES
from misc.hyperparameters import HyperParameters
from misc import utils
from models.transformer.encoder import TransformerEncoder
from models.softmax_output import SoftmaxOutputLayer, OutputLayer
from models.transformer_tagger import TransformerTagger
from models.transformer.train import Trainer
from criterion import NllLoss

import torch
experiment_name = None
PREFERENCES.defaults(
    data_root='./data/conll2003',
    data_train='eng.train.txt',
    data_validation='eng.testa.txt',
    data_test='eng.testb.txt',
    early_stopping='highest_5_F1'
)

hyper_parameters = HyperParameters(
    learning_rate_type='noam',
    learning_rate = 0.02,
    optim_adam_beta1=0.01,
    optim_adam_beta2=0.01)
experiment_name = utils.create_loggers(experiment_name=experiment_name)

conll2003 = conll2003_dataset('ner', 100,
                              root=PREFERENCES.data_root,
                              train_file=PREFERENCES.data_train,
                              validation_file=PREFERENCES.data_validation,
                              test_file=PREFERENCES.data_test)

num_units = 200


# 10 words with a 100-length embedding
target_vocab = conll2003['vocabs'][0]
target_size = len(target_vocab)

loss = NllLoss(target_size)
# transformer = GoogleTransformer(True, target_size, target_size, num_units, 2, 2, 512, 0.1)
transformer = TransformerEncoder(conll2003['embeddings'][0],
                                 n_enc_blocks=1,
                                 n_head=1,
                                 d_model=num_units,
                                 d_k=num_units,
                                 d_v=num_units)
tagging_softmax = SoftmaxOutputLayer(num_units, target_size)
model = TransformerTagger(transformer, tagging_softmax)
adam = torch.optim.Adam(model.parameters())
trainer = Trainer(model,
                    loss,
                    adam,
                    hyper_parameters,
                    conll2003['iters'],
                    -1,
                    experiment_name,
                    enable_tensorboard=True,
                    dummy_input=conll2003['dummy_input'])
trainer.train(4)
