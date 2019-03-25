import matplotlib
import copy
import logging

from data.data_loader import Dataset

from misc.preferences import PREFERENCES
from misc.run_configuration import get_default_params, randomize_params, OutputLayerType, hyperOpt_goodParams, elmo_params
from misc import utils

from optimizer import get_optimizer
from criterion import NllLoss, LossCombiner

from models.transformer.encoder import TransformerEncoder
from models.jointAspectTagger import JointAspectTagger
from trainer.train import Trainer
import pprint

# PREFERENCES.defaults(
# 	data_root='./data/data/germeval2017',
# 	data_train='train_v1.4.tsv',    
# 	data_validation='dev_v1.4.tsv',
# 	data_test='test_TIMESTAMP1.tsv',
# 	early_stopping='highest_5_F1'
# )
# from data.germeval2017 import germeval2017_dataset as dsl


from data.organic2019 import organic_dataset as dsl
from data.organic2019 import ORGANIC_TASK_ALL, ORGANIC_TASK_ENTITIES, ORGANIC_TASK_ATTRIBUTES
PREFERENCES.defaults(
    data_root='./data/data/organic2019',
    data_train='train.csv',    
    data_validation='validation.csv',
    data_test='test.csv',
    early_stopping='highest_5_F1'
)

def load(hp, logger):
	dataset = Dataset(
		ORGANIC_TASK_ATTRIBUTES,
		logger,
		hp,
		source_index=0,
		target_vocab_index=1,
		data_path=PREFERENCES.data_root,
		train_file=PREFERENCES.data_train,
		valid_file=PREFERENCES.data_validation,
		test_file=PREFERENCES.data_test,
		file_format='.csv',
		init_token=None,
		eos_token=None
	)
	dataset.load_data(dsl, verbose=True)
	return dataset

def load_model(dataset, hp, experiment_name):
	loss = LossCombiner(4, dataset.class_weights, NllLoss)
	transformer = TransformerEncoder(dataset.source_embedding,
									 hyperparameters=hp)
	model = JointAspectTagger(transformer, hp, 4, 20, dataset.target_names)
	optimizer = get_optimizer(model, hp)
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

experiment_name = 'organic2019'
use_cuda = True
experiment_name = utils.create_loggers(experiment_name=experiment_name)
logger = logging.getLogger(__name__)
logger.info('Current commit: ' + utils.get_current_git_commit())
hp = get_default_params(use_cuda, hyperOpt_goodParams)
hp.num_epochs = 20
hp.log_every_xth_iteration = -1
hp.adam_weight_decay = 1e-5
hp.pointwise_layer_size = 256
hp.n_enc_blocks = 2
hp.embedding_type = 'glove'
hp.language = 'en'
hp.use_spell_checkers = False

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
	#trainer.load_model(custom_path='C:\\Users\\felix\\OneDrive\\Studium\\Studium\\6. Semester\\MA\\Project\\ABSA-Transformer\\logs\\baseline\\20190323\\0\\checkpoints')
	#trainer.set_cuda(True)
	result = trainer.train(use_cuda=hp.use_cuda, perform_evaluation=True)
	model = result['model']
except Exception as err:
	logger.exception("Could not complete iteration because of " + str(err))
	print(f'Could not complete iteration because of {str(err)}')

# perform evaluation and log results
result = None
try:
	result = trainer.perform_final_evaluation(use_test_set=True, verbose=True)
except Exception as err:
	logger.exception("Could not complete iteration evaluation for it " + str(err))
	print(f'Could not complete iterationevaluation because of {str(err)}')




#trainer.load_model()

#trainer.set_cuda(True)
#result_labels = trainer.classify_sentence('Die Bahn preise sind sehr billig')

#result = trainer.train(use_cuda=hp.use_cuda, perform_evaluation=False)
#trainer.perform_final_evaluation(False)

#evaluation_results = trainer.perform_final_evaluation()
print('Exit')