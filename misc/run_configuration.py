from misc.utils import get_class_variable_table

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

                self.log_every_xth_iteration = kwargs['log_every_xth_iteration']
                self.num_epochs = num_epochs

                self.embedding_type = kwargs['embedding_type']
                self.embedding_name = kwargs['embedding_name']
                self.embedding_dim = kwargs['embedding_dim']

                self.language = kwargs['language']
                self.use_stop_words = kwargs['use_stop_words']
                self.seed = 42

        def __str__(self):
                return get_class_variable_table(self, 'Hyperparameters')



def get_default_params() -> RunConfiguration:
    return RunConfiguration(
        batch_size=100,
        model_size=256,
        learning_rate_type='noam',
        learning_rate=0,
        learning_rate_factor=2,
        learning_rate_warmup=4800,
        optim_adam_beta1=0.9,
        optim_adam_beta2=0.98,
        early_stopping=5,
        num_epochs=10,
        num_encoder_blocks=2,
        num_heads=3,
        att_d_k=100,
        att_d_v=100,
        dropout_rate=0.1,
        pointwise_layer_size=2048,
        log_every_xth_iteration=200,
        embedding_type='glove',
        embedding_dim=300,
        embedding_name='6B',
        language='en',
        use_stop_words=True,
        use_cuda=False
    )

