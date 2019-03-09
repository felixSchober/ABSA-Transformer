import numpy as np
from hyperopt import hp

def hp_bool(name):
    return hp.choice(name, [False, True])