import logging
import torch

#from tqdm.autonotebook import tqdm

from data.data_loader import Dataset
from data.germeval2017 import germeval2017_dataset

from misc.preferences import PREFERENCES
from misc.visualizer import *
from misc.run_configuration import get_default_params, randomize_params, LearningSchedulerType
from misc import utils

from optimizer import get_default_optimizer
from criterion import NllLoss, LossCombiner

from models.transformer.encoder import TransformerEncoder
from models.jointAspectTagger import JointAspectTagger
from models.transformer.train import Trainer
import pprint

import torchtext
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

PREFERENCES.defaults(
    data_root='./data/germeval2017',
    data_train='train_v1.4.tsv',    
    data_validation='dev_v1.4.tsv',
    data_test='test_TIMESTAMP1.tsv',
    early_stopping='highest_5_F1'
)
def load(hp, logger):
    dataset = Dataset(
        'germeval',
        logger,
        hp,
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
    dataset.load_data(germeval2017_dataset, verbose=False)
    return dataset

def load_model(dataset, hp, experiment_name):
    loss = LossCombiner(4, dataset.class_weights, NllLoss)
    transformer = TransformerEncoder(dataset.source_embedding,
                                     hyperparameters=hp)
    model = JointAspectTagger(transformer, hp, 4, 20, dataset.target_names)
    optimizer = get_default_optimizer(model, hp)
    trainer = Trainer(
                        model,
                        loss,
                        optimizer,
                        hp,
                        dataset,
                        experiment_name,
                        enable_tensorboard=False,
                        verbose=False)
    return trainer

def produce_test_gold_labels(iterator: torchtext.data.Iterator, dataset: Dataset, filename='gold_labels.xml'):

    fields = dataset.fields
    with torch.no_grad():
        iterator.init_epoch()
        
        tree = ET.ElementTree()
        root = ET.Element('Documents')

        for batch in iterator:
            doc_id, comment, relevance, aspect_sentiment, general_sentiment = batch.id, batch.comments, batch.relevance, batch.aspect_sentiments, batch.general_sentiments
            doc_id = fields['id'].reverse(doc_id.unsqueeze(1))
            comment = fields['comments'].reverse(comment)
            relevance = ['false' if r == 0 else 'true' for r in relevance]
            general_sentiment = fields['general_sentiments'].reverse(general_sentiment.unsqueeze(1))
            aspect_sentiment = fields['aspect_sentiments'].reverse(aspect_sentiment, detokenize=False)

            for i in range(len(doc_id)):
                docuement_elem = ET.SubElement(root, 'Document', {'id': doc_id[i]})

                rel_field = ET.SubElement(docuement_elem, 'relevance')
                rel_field.text = relevance[i]

                sen_field = ET.SubElement(docuement_elem, 'sentiment')
                sen_field.text = general_sentiment[i]

                text_field = ET.SubElement(docuement_elem, 'text')
                text_field.text = comment[i]

                # don't add aspects if field not relevant
                if relevance[i] == 'false':
                    continue
                options_elem = ET.SubElement(docuement_elem, 'Opinions')

                # add aspects
                for sentiment, a_name in zip(aspect_sentiment[i], dataset.target_names):
                    if sentiment == 'n/a':
                        continue

                    asp_field = ET.SubElement(options_elem, 'Opinion', {
                        'category': a_name,
                        'target': sentiment
                    })

        #print(BeautifulSoup(ET.tostring(tree), "xml").prettify())
        tree._setroot(root)
        tree.write(filename, encoding='utf-8')

def write_evaluation_file(iterator: torchtext.data.Iterator, dataset: Dataset, trainer: Trainer, filename='prediction.xml'):
    fields = dataset.fields
    with torch.no_grad():
        iterator.init_epoch()
        
        tree = ET.ElementTree()
        root = ET.Element('Documents')

        for batch in iterator:
            doc_id, comment, relevance, aspect_sentiment, general_sentiment, padding = batch.id, batch.comments, batch.relevance, batch.aspect_sentiments, batch.general_sentiments, batch.padding
            doc_id = fields['id'].reverse(doc_id.unsqueeze(1))
            comment_decoded = fields['comments'].reverse(comment)
            relevance = ['false' if r == 0 else 'true' for r in relevance]
            general_sentiment = fields['general_sentiments'].reverse(general_sentiment.unsqueeze(1))

            source_mask = trainer.create_padding_masks(padding, 1)
            prediction = trainer.model.predict(comment, source_mask)
            aspect_sentiment = fields['aspect_sentiments'].reverse(prediction, detokenize=False)

            for i in range(len(doc_id)):
                docuement_elem = ET.SubElement(root, 'Document', {'id': doc_id[i]})

                rel_field = ET.SubElement(docuement_elem, 'relevance')
                rel_field.text = relevance[i]

                sen_field = ET.SubElement(docuement_elem, 'sentiment')
                sen_field.text = general_sentiment[i]

                text_field = ET.SubElement(docuement_elem, 'text')
                text_field.text = comment_decoded[i]

                # don't add aspects if field not relevant
                if relevance[i] == 'false':
                    continue
                options_elem = ET.SubElement(docuement_elem, 'Opinions')

                # add aspects
                for sentiment, a_name in zip(aspect_sentiment[i], dataset.target_names):
                    if sentiment == 'n/a':
                        continue

                    asp_field = ET.SubElement(options_elem, 'Opinion', {
                        'category': a_name,
                        'target': sentiment
                    })

        #print(BeautifulSoup(ET.tostring(tree), "xml").prettify())
        tree._setroot(root)
        tree.write(filename, encoding='utf-8')



PREFERENCES.defaults(
    data_root='./data/germeval2017',
    data_train='train_v1.4.tsv',    
    data_validation='dev_v1.4.tsv',
    data_test='test_TIMESTAMP1.tsv',
    early_stopping='highest_5_F1'
)
# experiment_name = utils.create_loggers(experiment_name='testing')
# logger = logging.getLogger(__name__)

# default_hp = get_default_params(False)

# logger.info(default_hp)
# print(default_hp)

# dataset = load(default_hp, logger)
# produce_test_gold_labels(dataset.test_iter, dataset)

experiment_name = 'HyperParameterTest'
use_cuda = True
experiment_name = utils.create_loggers(experiment_name=experiment_name)
logger = logging.getLogger(__name__)

hp = get_default_params(True)

hp.learning_rate_type = LearningSchedulerType.Noam
hp.n_enc_blocks = 2
hp.n_heads = 10
hp.d_k = 30
hp.d_v = 30
hp.dropout_rate = 0.06955106610
hp.pointwise_layer_size = 134
hp.clip_comments_to = 391
logger.info(hp)
print(hp)
dataset = load(hp, logger)
trainer = load_model(dataset, hp, experiment_name)
import os
path = os.path.join(os.getcwd(), 'logs', 'GoodResults')
print(path)
trainer.load_model(custom_path=path)
trainer.set_cuda(True)
write_evaluation_file(dataset.test_iter, dataset, trainer)