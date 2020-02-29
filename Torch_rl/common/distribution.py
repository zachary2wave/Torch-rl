import torch
import numpy as np

def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        return DiagGaussianPd(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPd(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPd(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPd(ac_space.n)
    else:
        raise NotImplementedError

class Pd(object):
    """
    A particular probability distribution
    """
    def build(self,actor):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError



    def logp(self, x):
        return - self.neglogp(x)



class DiagGaussianPd(Pd):
    def __init__(self, shape):
        self.shape = shape

    def build(self, actor):
        self.actor = actor
        layer_infor = []
        for name, param in self.named_parameters():
            if "weight" in name:
                layer_infor.append(list(param.size()))
        self.output_layer = layer_infor[-1][0]

    def sample(self, output):
        if self.output_layer == self.shape:
            self.mean = output
            self.logstd = torch.ones_like(output)
        elif self.output_layer == self.shape*2:
            self.mean = output[:self.shape]
            self.logstd = output[self.shape:]
        self.std = torch.exp(self.logstd)
        return torch.normal(self.mean, self.std)

    def neglogp(self, x):
        return 0.5 * torch.sum(torch.pow((x - self.mean) / self.std),2, dim =-1) \
        + torch.tensor(0.5 * np.log(2.0 * np.pi) * list(x.size())) \
        + torch.sum(self.std, dim =-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return torch.sum(other.logstd - self.logstd + (self.std.pow(2) + torch.pow(self.mean - other.mean,2)) / (2.0 * other.std.pow(2)) - 0.5, axis=-1)

    def entropy(self):
        return torch.sum(self.logstd + torch.tensor( .5 * np.log(2.0 * np.pi * np.e)), axis=-1)



class CategoricalPd(Pd):


    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
class MultiCategoricalPd(Pd):
    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

class BernoulliPd(Pd):
    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError