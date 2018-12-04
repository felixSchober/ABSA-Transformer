class HyperParameters(object):

    learning_rate_type: str
    optim_adam_beta1: float
    optim_adam_beta2: float
    learning_rate: float

    def __init__(self,
                learning_rate_type: str,
                learning_rate: float,
                optim_adam_beta1: float,
                optim_adam_beta2: float):
        assert learning_rate_type == 'noam' or learning_rate_type == 'exp'

        self.learning_rate_type = learning_rate_type
        self.learning_rate = learning_rate

        self.optim_adam_beta1 = optim_adam_beta1
        self.optim_adam_beta2 = optim_adam_beta2