#
# Genetic algorithm implementation
#
from .base import Optimizer
# from .lib.genetic_algorithm import GeneticAlgorithm   # doing random search when mutation happens
from .lib.genetic_algorithm import DriftAlgorithm as GeneticAlgorithm  # drift around the original value when mutation happens

class GeneticOptimizer(Optimizer):
    def target_function(self, code):
        """
        target function to be optimized by bayesian optimizer.
        Param:
            params: ME parameters in a specific order (returned by ParamManager.encode())
        Return:
            evaluation metric under params
        """
        self.ensembler.param.decode(code)
        return self.forward()

    def optimize(self, mode):
        """ search for the optimal ensemble parameters that optimize the target metric """
        assert mode in ['maximize', 'minimize']
        def target_func(code):
            if mode == 'maximize':
                return self.target_function(code)
            else:
                return -self.target_function(code)

        _, bounds = self.ensembler.param.encode()
        genetic_alg = GeneticAlgorithm(target_func, bounds)

        code = genetic_alg.evolve()
        param = self.ensembler.param.clone()
        param.decode(code)
        return param