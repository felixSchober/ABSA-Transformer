class HyperParameters(object):

    # GENERAL MODEL
    model_size: int
    batch_size: int

    # OPTIMIZER
    learning_rate_type: str
    learning_rate_warmup: float
    learning_rate_factor: float
    optim_adam_beta1: float
    optim_adam_beta2: float
    learning_rate: float

    # TRAINING
    early_stopping: int

    def __init__(self,
                model_size: int,
                batch_size: int,
                learning_rate_type: str,
                learning_rate: float,
                learning_rate_factor: float,
                learning_rate_warmup: float,
                optim_adam_beta1: float,
                optim_adam_beta2: float,
                early_stopping: int):

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

def get_default_params() -> HyperParameters:
    return HyperParameters(
        batch_size=100,
        model_size=256,
        learning_rate_type='noam',
        learning_rate=0,
        learning_rate_factor=2,
        learning_rate_warmup=4800,
        optim_adam_beta1=0.9,
        optim_adam_beta2=0.98,
        early_stopping=5
    )