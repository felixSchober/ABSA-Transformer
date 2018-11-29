import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxOutputLayer(nn.Module):

    def __init__(self, hidden_size: int, output_size: int):
        """Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification for example
        
        Arguments:
            hidden_size {int} -- output of the transformer encoder (d_model)
            output_size {int} -- number of classes
        """
        super(SoftmaxOutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        logits = self.output_projection(x)
        probs = F.softmax(logits, -1)

        return probs

    def predict(self, x):
        probs = self.forward(x)
        _, predictions = torch.max(probs, dim=-1)
        return predictions

class OutputLayer(nn.Module):

    def __init__(self, hidden_size: int, output_size: int):
        """Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification for example
        
        Arguments:
            hidden_size {int} -- output of the transformer encoder (d_model)
            output_size {int} -- number of classes
        """
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        logits = self.output_projection(x)
        return logits

    def predict(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, -1)
        _, predictions = torch.max(probs, dim=-1)
        return predictions