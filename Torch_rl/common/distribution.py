import torch
import numpy as np

def make_pdtype(ac_space,actor):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        shape = ac_space.shape[0]
        layer_infor = []
        for name, param in actor.named_parameters():
            if "weight" in name:
                layer_infor.append(list(param.size()))
        output_layer = layer_infor[-1][0]
        return DiagGaussianPd_type(shape, output_layer)
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPd(ac_space.n, actor)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPd(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPd(ac_space.n, actor)
    else:
        raise NotImplementedError


class DiagGaussianPd_type():
    def __init__(self, shape, output_layer):
        self.shape = shape
        self.output_layer = output_layer

    def __call__(self, output, *args, **kwargs):
        if self.output_layer == self.shape:
            self.mean = output
            self.logstd = torch.ones_like(output)
        elif self.output_layer == self.shape*2:
            self.mean = torch.index_select(output.cpu(), -1, torch.arange(0, self.shape))
            self.logstd = torch.index_select(output.cpu(), -1, torch.arange(self.shape, self.shape*2))
        self.std = torch.exp(self.logstd)
        return DiagGaussianPd(self.mean, self.std)



class Pd(object):
    """
    A particular probability distribution
    """
    def log_prob(self, x):
        if len(x.shape)>1:
            return torch.sum(self.pd.log_prob(x),dim=-1,keepdim=True)
        else:
            return self.pd.log_prob(x)
    def sample(self):
        return self.pd.sample()

    def neglogp(self, x):
        return -self.pd.log_prob(x)

    def kl(self, other):
        return torch.distributions.kl.kl_vergence(self.pd, other)

    def entropy(self):
        return self.pd.entropy()



class DiagGaussianPd(Pd):
    def __init__(self, mean, std):
        from torch.distributions import Normal
        self.pd = Normal(mean, std)


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