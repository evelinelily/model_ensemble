#
# Optimizer class, designed for searching for optimal ensemble parameters
#

# import os, sys
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ensembler import Ensembler as EnsemblerFamily
from evaluator import Evaluator as EvaluatorFamily


class Optimizer(object):

    def __init__(self, ensembler, evaluator):
        """
        Class initializer:
        Arguments:
            ensembler: Ensembler object, which already loads predictions to be fused
            evaluator: Evaluator object
        """
        print(ensembler)
        print(EnsemblerFamily)
        print(evaluator)
        print(EvaluatorFamily)
        assert isinstance(ensembler, EnsemblerFamily)
        assert isinstance(evaluator, EvaluatorFamily)

        self.ensembler = ensembler
        self.param = self.ensembler.param
        self.evaluator = evaluator

    def forward(self):
        """ run model ensemble and evaluation in a row, return the target metric """
        fused_predictions = self.ensembler.fuse()
        metric = self.evaluator.compute_metric(fused_predictions)
        return metric

    def optimize(self, mode):
        raise NotImplementedError

    def maximize(self):
        return self.optimize(mode='maximize')

    def minimize(self):
        return self.optimize(mode='minimize')