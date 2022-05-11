#
# Optimizer class, designed for searching for optimal ensemble parameters
#
from .base import Optimizer

class RandomSearchOptimizer(Optimizer):
    def optimize(self, mode):
        """ search for the optimal ensemble parameters that optimize the target metric """
        assert mode in ['maximize', 'minimize']
        max_metric = self.forward()
        opt_param = self.ensembler.param.clone()
        for ix in range(1000):
            print("[{}] max metric: {}".format(ix, max_metric))
            self.ensembler.param.randomly_sample()
            metric = self.forward()
            condition = metric > max_metric if mode == 'maximize' else metric < max_metric
            if condition:
                opt_param = self.ensembler.param.clone()
                max_metric = metric
                opt_param.dump()
        return opt_param