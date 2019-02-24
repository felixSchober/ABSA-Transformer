from misc.utils import get_class_variable_table
import random

class RunConfiguration(object):

        def __init__(self,
                model_size: int,
                batch_size: int,
                learning_rate_type: str,
                learning_rate: float,
                learning_rate_factor: float,
                learning_rate_warmup: float,
                optim_adam_beta1: float,
                optim_adam_beta2: float,
                early_stopping: int,
                num_epochs:int,
                **kwargs):

                assert model_size > 0
                assert learning_rate_type == 'noam' or learning_rate_type == 'exp'
                assert batch_size > 0
                assert early_stopping == -1 or early_stopping > 0

                self.batch_size = batch_size
                self.model_size = model_size
                self.learning_rate_type = learning_rate_type
                self.learning_rate = learning_rate
                self.learning_rate_warmup = learning_rate_warmup
                self.learning_rate_factor = learning_rate_factor
                self.optim_adam_beta1 = optim_adam_beta1
                self.optim_adam_beta2 = optim_adam_beta2

                self.early_stopping = early_stopping

                self.use_cuda = kwargs['use_cuda']

                self.n_enc_blocks = kwargs['num_encoder_blocks']
                self.n_heads = kwargs['num_heads']
                self.d_k = kwargs['att_d_k']
                self.d_v = kwargs['att_d_v']
                self.dropout_rate = kwargs['dropout_rate']
                self.pointwise_layer_size = kwargs['pointwise_layer_size']

                self.output_layer_type = kwargs['output_layer_type']

                self.output_conv_num_filters = kwargs['output_conv_num_filters']
                self.output_conv_kernel_size = kwargs['output_conv_kernel_size']
                self.output_conv_stride = kwargs['output_conv_stride']
                self.output_conv_padding = kwargs['output_conv_padding']

                self.log_every_xth_iteration = kwargs['log_every_xth_iteration']
                self.num_epochs = num_epochs

                self.embedding_type = kwargs['embedding_type']
                self.embedding_name = kwargs['embedding_name']
                self.embedding_dim = kwargs['embedding_dim']
                self.clip_comments_to = kwargs['clip_comments_to']

                self.language = kwargs['language']
                self.use_stop_words = kwargs['use_stop_words']
                self.seed = 42

        def __str__(self):
                return get_class_variable_table(self, 'Hyperparameters')

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

        if 'transformer_config' in param_dict_range:
                transformer_config = param_dict_range['transformer_config']

                if 'transformer_heads' in transformer_config:                        
                        config.n_heads = random.choice(transformer_config['transformer_heads'])

                        # based on num_heads choose the d_v, d_k, d_q sizes
                        # make sure that it will be a valid size
                        assert(config.model_size % config.n_heads == 0, f"number of heads {config.n_heads} is not a valid number of heads for model size {config.model_size}.")

                        config.d_k = config.model_size // config.n_heads
                        config.d_v = config.model_size // config.n_heads

        return config


def get_default_params(use_cuda: bool = False) -> RunConfiguration:
    return RunConfiguration(
        batch_size=12,
        model_size=300,
        learning_rate_type='noam',
        learning_rate=0,
        learning_rate_factor=2,
        learning_rate_warmup=4800,
        optim_adam_beta1=0.9,
        optim_adam_beta2=0.98,
        early_stopping=5,
        num_epochs=1,
        num_encoder_blocks=3,
        num_heads=6,
        att_d_k=50,
        att_d_v=50,
        dropout_rate=0.1,
        pointwise_layer_size=2048,
        log_every_xth_iteration=-1,
        embedding_type='fasttext',
        embedding_dim=300,
        embedding_name='6B',
        language='de',
        use_stop_words=True,
        use_cuda=use_cuda,
        clip_comments_to=100,
        output_layer_type='conv',
        output_conv_num_filters=300,
        output_conv_kernel_size=5,
        output_conv_stride=1,
        output_conv_padding=0
    )

