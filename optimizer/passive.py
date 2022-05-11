from .base import Optimizer

class PassiveOptimizer(Optimizer):
    """ Optimizer that does nothing when do optimization: use the initial parameters """
    def optimize(self, mode):
        return self.param.clone()