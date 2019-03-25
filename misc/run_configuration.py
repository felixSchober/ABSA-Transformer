from misc.utils import get_class_variable_table, set_seeds
import random
from enum import Enum
from typing import Dict

class OutputLayerType(Enum):
        LinearSum = 1
        Convolutions = 2

class LearningSchedulerType(Enum):
		Noam = 1
		Exponential = 2
		NoSchedule = 3

class OptimizerType(Enum):
		Adam = 1
		SGD = 2
		RMS_PROP = 3
		AdaBound = 4

default_params = {
	'model_size': 300,
	'batch_size': 12,
	'learning_rate_scheduler_type': LearningSchedulerType.Noam,
	'learning_rate_scheduler': {
		'noam_learning_rate_warmup': 6000,
		'noam_learning_rate_factor': 0.8
	},
	'optimizer_type':  OptimizerType.Adam,
	'optimizer':  {
		'learning_rate': 1e-5,
		'adam_beta1': 0.9,
		'adam_beta2': 0.98,
		'adam_eps': 1e-9
	},
	'early_stopping': 5,
	'num_epochs': 25,
	'num_encoder_blocks': 2,
	'num_heads': 6,
	'att_d_k': 50,
	'att_d_v': 50,
	'dropout_rate': 0.1,
	'pointwise_layer_size': 128,
	'log_every_xth_iteration': -1,
	'embedding_type': 'fasttext',
	'embedding_dim': 300,
	'embedding_name': '6B',
	'language': 'de',
	'use_stop_words': True,
	'clip_comments_to': 100,
	'output_layer_type': OutputLayerType.LinearSum,
	'output_layer':  {
		'output_conv_num_filters': 300,
		'output_conv_kernel_size': 5,
		'output_conv_stride': 1,
		'output_conv_padding': 0
	},
	'output_dropout_rate': 0.2
}

hyperOpt_goodParams = {
	'output_layer_type': OutputLayerType.LinearSum,
	'embedding_type': 'fasttext',
	'learning_rate_scheduler_type': LearningSchedulerType.Noam,
	'learning_rate_scheduler': {
		'noam_learning_rate_warmup': 8000,
		'noam_learning_rate_factor': 1.418
	},
	'optimizer_type':  OptimizerType.Adam,
	'optimizer':  {
		'learning_rate': 7.2e-5,
		'adam_beta1': 0.81,
		'adam_beta2': 0.7173,
		'adam_eps': 0.0008140
	},
	'num_encoder_blocks': 2,
	'num_heads': 1,
	'att_d_k': 300,
	'att_d_v': 300,
	'dropout_rate': 0.302424,
	'pointwise_layer_size': 405,
	'output_dropout_rate': 0.79602089766246,
	'clip_comments_to': 113,
	'harmonize_bahn': True,
	'model_size': 300

}

elmo_params = {
	'output_layer_type': OutputLayerType.LinearSum,
	'embedding_type': 'elmo',
	'learning_rate_scheduler_type': LearningSchedulerType.Noam,
	'learning_rate_scheduler': {
		'noam_learning_rate_warmup': 8000,
		'noam_learning_rate_factor': 1.418
	},
	'optimizer_type':  OptimizerType.Adam,
	'optimizer':  {
		'learning_rate': 7.2e-5,
		'adam_beta1': 0.81,
		'adam_beta2': 0.7173,
		'adam_eps': 0.0008140
	},
	'num_encoder_blocks': 2,
	'num_heads': 8,
	'att_d_k': 128,
	'att_d_v': 128,
	'dropout_rate': 0.302424,
	'pointwise_layer_size': 405,
	'output_dropout_rate': 0.79602089766246,
	'clip_comments_to': 113,
	'harmonize_bahn': True,
	'model_size': 1024,
	'att_d_k': 128,
	'att_d_v': 128,
}

class RunConfiguration(object):

		def __init__(self,
				use_cuda: bool,
				model_size: int,
				early_stopping: int,
				num_epochs:int,
				log_every_xth_iteration: int,
				output_layer_type: OutputLayerType,
				learning_rate_scheduler_type: LearningSchedulerType,
				optimizer_type: LearningSchedulerType,
				language: str,
				**kwargs):
			self.kwargs = kwargs
			assert model_size > 0
			assert kwargs['batch_size'] > 0
			assert early_stopping == -1 or early_stopping > 0

			self.model_size = model_size
			self.early_stopping = early_stopping
			self.use_cuda = use_cuda

			self.batch_size = self._get_default('batch_size', cast_int=True)

			# types
			self.learning_rate_scheduler_type = learning_rate_scheduler_type
			self.output_layer_type = output_layer_type
			self.optimizer_type = optimizer_type

			# LEARNING RATE SCHEDULER
			self.learning_rate = kwargs['optimizer']['learning_rate']


			# - NOAM
			if learning_rate_scheduler_type == LearningSchedulerType.Noam:
				p = kwargs['learning_rate_scheduler']		
				
				self.noam_learning_rate_warmup = self._get_default('noam_learning_rate_warmup', section=p, cast_int=True)
				self.noam_learning_rate_factor = p['noam_learning_rate_factor']

			# - Exponential
			elif learning_rate_scheduler_type == LearningSchedulerType.Exponential:
				p = kwargs['learning_rate_scheduler']	
				self.exponentiallr_gamma = self._get_default('exponentiallr_gamma', 0.9, section=p)


			# OPTIMIZER
			# - ADAM
			if optimizer_type == OptimizerType.Adam:
				p = kwargs['optimizer']		

				self.adam_beta1 = p['adam_beta1']
				self.adam_beta2 = p['adam_beta2']
				self.adam_eps = p['adam_eps']
				self.adam_weight_decay = self._get_default('adam_weight_decay', 0, section=p)
				self.adam_amsgrad = self._get_default('adam_amsgrad', False, section=p)

			# - SGD
			elif optimizer_type == OptimizerType.SGD:
				p = kwargs['optimizer']		

				self.sgd_momentum = p['sgd_momentum']
				self.sgd_weight_decay = p['sgd_weight_decay']
				self.sgd_dampening = self._get_default('sgd_dampening', 0, section=p)
				self.sgd_nesterov = p['sgd_nesterov']

			# - AdaBound
			elif optimizer_type == OptimizerType.AdaBound:
				p = kwargs['optimizer']		

				self.adabound_finallr = self._get_default('adabound_finallr', 0.1, section=p)

			# - RMSprop
			elif optimizer_type == OptimizerType.RMS_PROP:
				p = kwargs['optimizer']		

				self.rmsprop_momentum = self._get_default('rmsprop_momentum', 0, section=p)
				self.rmsprop_alpha = self._get_default('rmsprop_alpha', 0.99, section=p)
				self.rmsprop_eps = self._get_default('rmsprop_eps', 1e-8, section=p)
				self.rmsprop_centered = self._get_default('rmsprop_centered', False, section=p)
				self.rmsprop_weight_decay = self._get_default('rmsprop_weight_decay', 0, section=p)

			# Transformer

			self.use_bias = self._get_default('transformer_use_bias', False)
			self.n_enc_blocks = self._get_default('num_encoder_blocks', cast_int=True)
			self.n_heads = self._get_default('num_heads', cast_int=True)
			self.d_k = self.model_size // self.n_heads
			self.d_v = self.model_size // self.n_heads
			self.dropout_rate = kwargs['dropout_rate']
			self.pointwise_layer_size = self._get_default('pointwise_layer_size', cast_int=True)

			# Output Layer
			self.last_layer_dropout = kwargs['output_dropout_rate']

			# - Convolutions:	
			if output_layer_type == OutputLayerType.Convolutions:
				p = kwargs['output_layer']		
				self.output_conv_num_filters = self._get_default('output_conv_num_filters', section=p, cast_int=True)
				self.output_conv_kernel_size = self._get_default('output_conv_kernel_size', section=p, cast_int=True)
				self.output_conv_stride = self._get_default('output_conv_stride', section=p, cast_int=True)
				self.output_conv_padding = self._get_default('output_conv_padding', section=p, cast_int=True)

			self.log_every_xth_iteration = log_every_xth_iteration
			self.num_epochs = num_epochs

			# - Embedding
			assert kwargs['embedding_type'] in ['glove', 'fasttext', 'elmo']
			self.embedding_type = kwargs['embedding_type']
			self.embedding_name = kwargs['embedding_name']
			self.embedding_dim = kwargs['embedding_dim']
			self.clip_comments_to = self._get_default('clip_comments_to', cast_int=True)

			self.language = language

			# data loading
			self.use_stop_words = self._get_default('use_stop_words', False)
			self.use_stemming = self._get_default('use_stemming', False)
			self.harmonize_bahn = self._get_default('harmonize_bahn', False)
			self.use_spell_checkers = self._get_default('use_spell_checkers', False)
			self.replace_url_tokens = self._get_default('replace_url_tokens', False)
			self.use_text_cleaner = self._get_default('use_text_cleaner', False)
			
			self.seed = 42			
			set_seeds(self.seed)

		def _get_default(self, key: str, default=None, section: Dict = None, cast_int: bool=False):
			if section is None:
				section = self.kwargs
			if key in section:
				v = section[key]
				if cast_int:
					v = int(v)
				return v
			return default

		def __str__(self):
			return get_class_variable_table(self, 'Hyperparameters')

		def __eq__(self, other):
			return self.__dict__ == other.__dict__

		def run_equals(self, other):
			# those are the keys that are necessary to run a saved model
			keys = ['model_size', 'use_cuda', 'n_enc_blocks', 'n_heads', 'd_k', 'd_v', 'dropout_rate',
			'pointwise_layer_size', 'output_layer_type', 'output_conv_num_filters', 'output_conv_kernel_size', 
			'output_conv_stride', 'output_conv_padding', 'embedding_dim', 'clip_comments_to', 'optimizer_type', 
			'embedding_type', 'output_layer_type', 'transformer_use_bias', 'output_dropout_rate']

			for k in keys:
				if not k in self.__dict__ and not k in other.__dict__:
					continue

				if not k in self.__dict__ or not k in self.__dict__:
					return False

				if self.__dict__[k] != other.__dict__[k]:
					return False
			return True

def randomize_params(config, param_dict_range) -> RunConfiguration:

        if 'batch_size' in param_dict_range:
                ranges = param_dict_range['batch_size']
                config.batch_size = random.randint(ranges[0], ranges[1])

        if 'num_encoder_blocks' in param_dict_range:
                ranges = param_dict_range['num_encoder_blocks']
                config.n_enc_blocks = random.randint(ranges[0], ranges[1])

        if 'pointwise_layer_size' in param_dict_range:
                ranges = param_dict_range['pointwise_layer_size']
                config.pointwise_layer_size = random.randint(ranges[0], ranges[1])

        if 'output_conv_num_filters' in param_dict_range:
                ranges = param_dict_range['output_conv_num_filters']
                config.output_conv_num_filters = random.randint(ranges[0], ranges[1])

        if 'output_conv_kernel_size' in param_dict_range:
                ranges = param_dict_range['output_conv_kernel_size']
                config.output_conv_kernel_size = random.randint(ranges[0], ranges[1])

        if 'output_conv_stride' in param_dict_range:
                ranges = param_dict_range['output_conv_stride']
                config.output_conv_stride = random.randint(ranges[0], ranges[1])

        if 'output_conv_padding' in param_dict_range:
                ranges = param_dict_range['output_conv_padding']
                config.output_conv_padding = random.randint(ranges[0], ranges[1])

        if 'output_layer_type' in param_dict_range:
                types = param_dict_range['output_layer_type']
                config.output_layer_type = random.choice(types)

        if 'clip_comments_to' in param_dict_range:
                ranges = param_dict_range['clip_comments_to']
                config.clip_comments_to = random.randint(ranges[0], ranges[1])

        if 'learning_rate' in param_dict_range:
                ranges = param_dict_range['learning_rate']
                config.learning_rate = random.uniform(ranges[0], ranges[1])

        if 'learning_rate_factor' in param_dict_range:
                ranges = param_dict_range['learning_rate_factor']
                config.learning_rate_factor = random.uniform(ranges[0], ranges[1])

        if 'learning_rate_warmup' in param_dict_range:
                ranges = param_dict_range['learning_rate_warmup']
                config.learning_rate_warmup = random.uniform(ranges[0], ranges[1])

        if 'optim_adam_beta1' in param_dict_range:
                ranges = param_dict_range['optim_adam_beta1']
                config.optim_adam_beta1 = random.uniform(ranges[0], ranges[1])

        if 'optim_adam_beta2' in param_dict_range:
                ranges = param_dict_range['optim_adam_beta2']
                config.optim_adam_beta2 = random.uniform(ranges[0], ranges[1])

        if 'dropout_rate' in param_dict_range:
                ranges = param_dict_range['dropout_rate']
                config.dropout_rate = random.uniform(ranges[0], ranges[1])

        if 'last_layers_dropout_rate' in param_dict_range:
                ranges = param_dict_range['last_layers_dropout_rate']
                config.last_layers_dropout_rate = random.uniform(ranges[0], ranges[1])

        if 'transformer_config' in param_dict_range:
                transformer_config = param_dict_range['transformer_config']

                if 'transformer_heads' in transformer_config:                        
                        config.n_heads = random.choice(transformer_config['transformer_heads'])

                        # based on num_heads choose the d_v, d_k, d_q sizes
                        # make sure that it will be a valid size
                        assert(config.model_size % config.n_heads == 0, f'number of heads {config.n_heads} is not a valid number of heads for model size {config.model_size}.')

                        config.d_k = config.model_size // config.n_heads
                        config.d_v = config.model_size // config.n_heads

        return config

def from_hyperopt(hyperopt_params,
				use_cuda: bool,
				model_size: int,
				early_stopping: int,
				num_epochs:int,
				log_every_xth_iteration: int,
				language: str) -> RunConfiguration:
	rc = RunConfiguration(
		use_cuda,
		model_size,
		early_stopping,
		num_epochs,
		log_every_xth_iteration,
		hyperopt_params['output_layer']['type'],
		hyperopt_params['learning_rate_scheduler']['type'],
		hyperopt_params['optimizer']['type'],
		language,
		**hyperopt_params)
	return rc
	

def get_default_params(use_cuda: bool = False, overwrite: Dict = None) -> RunConfiguration:
	params = default_params
	if overwrite:
		params = {**default_params, **overwrite}

	return RunConfiguration(
		use_cuda,
		**params
	)