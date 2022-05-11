from .base import ParamManager
import numpy as np

class ClassificationParam(ParamManager):
    def initialize_weights(self, num_models, num_classes):
        """
        Initialize weights for distribution fusion.
        Arguments:
            num_models: #models to be ensembled
            num_classes: #classes in model predictions (aka. length of distribution)
        """
        assert isinstance(num_classes, int) and isinstance(num_models, int)
        if not hasattr(self, 'dist_weights'):  # can only be set once
            self.num_models  = num_models
            self.num_classes = num_classes
            self.dist_weights = .5 + np.zeros([num_models, num_classes])

    def __getitem__(self, iModel):
        assert 0 <= iModel < self.num_models
        return self.dist_weights[iModel]

    def dump(self):
        print("Ensemble Parameters:")
        for iModel, class_weights in enumerate(self.dist_weights):
            print("Model{} class weight: {}".format(iModel+1, class_weights.tolist()))

    def encode(self):
        """ 
        Encode all parameters to a vector, usually used by some optimizers (bayesian optimizer, genetic algorithm)
        Return:
            code: A list that enumerate all parameters in a fixed order
            bounds: A list of tuples (Nones if no boundaries are required) that lists the corresponding boundaries, like: [(low1, hight), (low2, high2), ...]
        """
        code = self.dist_weights.flatten().tolist()
        bounds = [(0,1)]*(self.num_models*self.num_classes)
        return code, bounds

    def decode(self, code):
        """
        Update parameters given code that generated by self.encode()[0]
        """
        assert self.check_code(code)
        self.dist_weights = np.array(code).reshape([self.num_models, self.num_classes])


class MultiClassParam(ClassificationParam):
    pass

class MultiLabelParam(ClassificationParam):
    pass