from data.conll import conll2003_dataset
from misc.preferences import PREFERENCES
from misc import utils
from models.transformer.googleTransformer import GoogleTransformer

from models.transformer.train import Trainer
from criterion import NllLoss

import torch
import torch.nn as nnp
import torch.nn.functional as F
import logging



PREFERENCES.defaults(
    data_root='./data/conll2003',
    data_train='eng.train.txt',
    data_validation='eng.testa.txt',
    data_test='eng.testb.txt',
    early_stopping='highest_5_F1'
)

conll2003 = conll2003_dataset('ner',  100,  
            root=PREFERENCES.data_root,
            train_file=PREFERENCES.data_train,
            validation_file=PREFERENCES.data_validation,
            test_file=PREFERENCES.data_test)

num_units = 512

utils.create_loggers()


# 10 words with a 100-lenght embedding
target_vocab = conll2003['vocabs'][2]
target_size = len(target_vocab)

loss = NllLoss(target_size)
adam = torch.optim.Adam()
transformer = GoogleTransformer(True, 3, 3, 512, 2, 2, 512, 0.1)
trainer = Trainer(transformer, loss, adam, None, conll2003['iters'], -1, 'test')
trainer.train(1)