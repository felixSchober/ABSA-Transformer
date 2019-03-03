import torch
import torchtext
import xml.etree.ElementTree as ET
from data.data_loader import Dataset
from bs4 import BeautifulSoup
import logging
from data.germeval2017 import germeval2017_dataset
from misc.preferences import PREFERENCES
from misc.run_configuration import get_default_params, randomize_params
from misc import utils
import pprint

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

def produce_test_gold_labels(iterator: torchtext.data.Iterator, dataset: Dataset):

    fields = dataset.fields
    with torch.no_grad():
        iterator.init_epoch()

        tree = ET.Element('Documents')

        for batch in iterator:
            doc_id, comment, relevance, aspect_sentiment, general_sentiment = batch.id, batch.comments, batch.relevance, batch.aspect_sentiments, batch.general_sentiments
            doc_id = fields['id'].reverse(doc_id.unsqueeze(1))
            comment = fields['comments'].reverse(comment)
            relevance = ['false' if r == 0 else 'true' for r in relevance]
            general_sentiment = fields['general_sentiments'].reverse(general_sentiment.unsqueeze(1))
            aspect_sentiment = fields['aspect_sentiments'].reverse(aspect_sentiment, detokenize=False)

            for i in range(len(doc_id)):
                docuement_elem = ET.SubElement(tree, 'Document', {'id': doc_id[i]})

                rel_field = ET.SubElement(docuement_elem, 'relevance')
                rel_field.text = relevance[i]

                sen_field = ET.SubElement(docuement_elem, 'sentiment')
                sen_field.text = general_sentiment[i]

                text_field = ET.SubElement(docuement_elem, 'text')
                text_field.text = comment[i]

                options_elem = ET.SubElement(docuement_elem, 'Opinions')

                # add aspects
                for sentiment, a_name in zip(aspect_sentiment[i], dataset.target_names):
                    if sentiment == 'n/a':
                        continue

                    asp_field = ET.SubElement(options_elem, 'Opinion', {
                        'category': a_name,
                        'target': sentiment
                    })

        print(BeautifulSoup(ET.tostring(tree), "xml").prettify())


PREFERENCES.defaults(
    data_root='./data/germeval2017',
    data_train='train_v1.4.tsv',    
    data_validation='dev_v1.4.tsv',
    data_test='test_TIMESTAMP1.tsv',
    early_stopping='highest_5_F1'
)
experiment_name = utils.create_loggers(experiment_name='testing')
logger = logging.getLogger(__name__)

default_hp = get_default_params(False)

logger.info(default_hp)
print(default_hp)

dataset = load(default_hp, logger)
produce_test_gold_labels(dataset.test_iter, dataset)