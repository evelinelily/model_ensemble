#
# Use bayesian optimization tool found in; https://github.com/fmfn/BayesianOptimization
# Installation: pip install bayesian-optimization
#

from .base import Optimizer
from bayes_opt import BayesianOptimization

class BayesianOptimizer(Optimizer):
    def target_function(self, **params_dict):
        """
        target function to be optimized by bayesian optimizer.
        Param:
            params: ME parameters in a specific order (returned by ParamManager.encode())
        Return:
            evaluation metric under params
        """
        code = [item[1] for item in sorted(params_dict.items(), key=lambda x: x[0])]
        self.ensembler.param.decode(code)
        return self.forward()


    def optimize(self, mode):
        """ search for the optimal ensemble parameters that optimize the target metric """
        assert mode in ['maximize', 'minimize']

        def target_func(**params_dict):
            if mode == 'maximize':
                return self.target_function(**params_dict)
            else:
                return -self.target_function(**params_dict)

        _, bounds = self.ensembler.param.encode()
        pbounds = {'param{:03d}'.format(ix): bnd for ix, bnd in enumerate(bounds)}

        optimizer = BayesianOptimization(
            f = target_func,
            pbounds = pbounds,
            random_state=1,
        )
        optimizer.maximize(init_points=2, n_iter=100)
        # recover code from the optimal results
        code = [item[1] for item in sorted(optimizer.max['params'].items(), key=lambda x: x[0])]
        param = self.ensembler.param.clone()
        param.decode(code)
        return param
