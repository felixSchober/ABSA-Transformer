import matplotlib
import copy
import logging

from data.data_loader import Dataset
from data.germeval2017 import germeval2017_dataset

from misc.preferences import PREFERENCES
from misc.run_configuration import get_default_params, randomize_params
from misc import utils

from optimizer import get_default_optimizer
from criterion import NllLoss, LossCombiner

from models.transformer.encoder import TransformerEncoder
from models.softmax_output import SoftmaxOutputLayerWithCommentWiseClass
from models.transformer_tagger import TransformerTagger
from models.jointAspectTagger import JointAspectTagger
from models.transformer.train import Trainer
import pprint

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
    dataset.load_data(germeval2017_dataset, verbose=True)
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
                        verbose=True)
    return trainer

experiment_name = 'Conv2dLayerTest'
use_cuda = True
experiment_name = utils.create_loggers(experiment_name=experiment_name)
logger = logging.getLogger(__name__)
utils.get_current_git_commit()
logger.info('Current commit: ' + utils.get_current_git_commit())
hp = get_default_params(use_cuda)
hp.num_epochs = 15
hp.log_every_xth_iteration = -1

logger.info(hp)
print(hp)


dataset_logger = logging.getLogger('data_loader')

logger.debug('Load dataset')
dataset = load(hp, dataset_logger)
logger.debug('dataset loaded')
logger.debug('Load model')
trainer = load_model(dataset, hp, experiment_name)
logger.debug('model loaded')

logger.debug('Begin training')
model = None
try:
    result = trainer.train(use_cuda=hp.use_cuda, perform_evaluation=True)
    model = result['model']
except Exception as err:
    logger.exception("Could not complete iteration because of " + str(err))
    print(f'Could not complete iteration because of {str(err)}')

# perform evaluation and log results
result = None
try:
    result = trainer.perform_final_evaluation(use_test_set=False, verbose=True)
except Exception as err:
    logger.exception("Could not complete iteration evaluation for it " + str(err))
    print(f'Could not complete iterationevaluation because of {str(err)}')




#trainer.load_model()

#trainer.set_cuda(True)
#result_labels = trainer.classify_sentence('Die Bahn preise sind sehr billig')

result = trainer.train(use_cuda=hyperparameters.use_cuda, perform_evaluation=False)
#trainer.perform_final_evaluation(False)

#evaluation_results = trainer.perform_final_evaluation()
print('Exit')