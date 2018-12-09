from __future__ import division
import numpy as np
import torch
import json
import os
import uuid
import errno
import logging
import sys
from torch.nn.modules.module import _addindent
import torch
import random

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_num_params(model: torch.nn.Module) -> int:
    """
    Computes number of trainable and non-trainable parameters
    From https://github.com/kolloldas/torchnlp/blob/master/torchnlp/common/train.py
    """
    sizes = [(np.array(p.data.size()).prod(), int(p.requires_grad)) for p in model.parameters()]
    return sum(map(lambda t: t[0]*t[1],sizes)), sum(map(lambda t: t[0]*(1 - t[1]),sizes))


def get_uuid():
    """ Generates a unique string id."""

    x = uuid.uuid1()
    return str(x)


def create_dir_if_necessary(path):
    """ Save way for creating a directory (if it does not exist yet).
    From http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary

    Keyword arguments:
    path -- path of the dir to check
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def check_if_file_exists(path):
    """ Checks if a file exists."""

    return os.path.exists(path)


def create_loggers(log_path=None, experiment_name=None, log_file_name='run.log'):

    if log_path is None:
        log_path = os.path.join(os.getcwd(), 'logs')
    
    create_dir_if_necessary(log_path)

    # if no experiment name specified - create unique name
    if experiment_name is None:
        import datetime
        now = datetime.datetime.now()
        experiment_name = '{0}_{1}'.format(now.strftime('%Y%m%d%H%M'), get_uuid())

    log_path = os.path.join(log_path, experiment_name)
    create_dir_if_necessary(log_path)
    print('Log path is ', log_path)

    create_dir_if_necessary(os.path.join(log_path, 'images'))
    create_dir_if_necessary(os.path.join(log_path, 'checkpoints'))


    # TODO: replace with logging config. See https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
    # setup logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=os.path.join(log_path, log_file_name),
                        filemode='w',
                        level=logging.DEBUG)

    # create loggers
    logger_main = logging.getLogger('pre_training')
    logger_main.setLevel(logging.DEBUG)

    logger_prediction = logging.getLogger('prediction')
    logger_prediction.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    prediction_handler = logging.StreamHandler(sys.stdout)
    prediction_handler.setLevel(logging.INFO)

    main_handler = logging.StreamHandler(sys.stdout)
    main_handler.setLevel(logging.DEBUG)

    # create formatters
    main_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    prediction_formatter = logging.Formatter('%(message)s')


    # add formatter to ch
    main_handler.setFormatter(main_formatter)
    prediction_handler.setFormatter(prediction_formatter)

    # add ch to logger
    logger_main.addHandler(main_handler)
    logger_prediction.addHandler(prediction_handler)

    return experiment_name


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""

    def summarize(model, num_params, show_weights=True, show_parameters=True):
        tmpstr = model.__class__.__name__ + ' (\n'
        for key, module in model._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr, num_params_rec = summarize(module, num_params, show_weights, show_parameters)
                num_params += num_params_rec
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            num_params += params
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr +=  ', parameters={}'.format(params)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'
        return tmpstr, num_params

    summary, num_params = summarize(model, 0, show_weights, show_parameters)

    summary = summary + '\n==================================\nTotal Number of parameters: {0}\n==================================\n'.format(num_params)
    return summary



def crop_to_square(image):
    """ Crops the square window of an image around the center."""

    if image is None:
        return None
    w, h = (image.shape[1], image.shape[0])
    w = float(w)
    h = float(h)

    # only crop images automatically if the aspect ratio is not bigger than 2 or not smaller than 0.5
    aspectRatio = w / h
    if aspectRatio > 3 or aspectRatio < 0.3:
        return None
    if aspectRatio == 1.0:
        return image

    # the shortest edge is the edge of our new square. b is the other edge
    a = min(w, h)
    b = max(w, h)

    # get cropping position
    x = (b - a) / 2.0

    # depending which side is longer we have to adjust the points
    # Height is longer
    if h > w:
        upperLeft = (0, x)
    else:
        upperLeft = (x, 0)
    cropW = cropH = a
    return crop_image(image, upperLeft[0], upperLeft[1], cropW, cropH)


def crop_image(image, x, y, w, h):
    """ Crops an image.

    Keyword arguments:
    image -- image to crop
    x -- upper left x-coordinate
    y -- upper left y-coordinate
    w -- width of the cropping window
    h -- height of the cropping window
    """

    # crop image using np slicing (http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python)
    image = image[y: y + h, x: x + w]
    return image


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)