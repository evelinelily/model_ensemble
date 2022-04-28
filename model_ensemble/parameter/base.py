#
# Classes that manage parameters for model ensemble
#
import copy, pickle, os
import numpy as np
from .utils import sample_range

class ParamManager(object):
    def encode(self):
        raise NotImplementedError

    def decode(self, code):
        raise NotImplementedError

    def clone(self):
        return copy.deepcopy(self)

    def dump(self):
        raise NotImplementedError

    def __getitem__(self, iModel):
        raise NotImplementedError

    def randomly_sample(self):
        """ randomly sample parameters in range """
        code, bounds = self.encode()
        assert len(code) == len(bounds)
        for ix, bnd in enumerate(bounds):
            code[ix] = sample_range(*bnd)
        self.decode(code)

    def check_code(self, code):
        """ Check if the input code compatible with the parameters """
        if not (isinstance(code, np.ndarray) or isinstance(code, list) or isinstance(code, tuple)):
            raise TypeError("the input type is not right")

        target_code, bounds = self.encode()

        # check length
        if len(code) != len(target_code):
            raise RuntimeError("Can not decode the given code: incompatible size")

        # check boundaries
        for val, bnd in zip(code, bounds):
            if bnd is None:
                continue
            if val < bnd[0] or val > bnd[1]:
                raise RuntimeError("Can not decode the given code: some value out of boundary")

        return True

    def pickle(self, pickle_path):
        print("Pickling ME hyper-parameters to file: {}".format(pickle_path))
        pickle_dir = os.path.dirname(pickle_path)
        if pickle_dir != '' and not os.path.isdir(pickle_dir):
            os.makedirs(pickle_dir)
        with open(pickle_path, 'wb') as fid:
            pickle.dump(self, fid)

    @classmethod
    def from_pickle(cls, pickle_path):
        assert os.path.exists(pickle_path), "Pickle file doesn't exist: {}".format(pickle_path)
        with open(pickle_path, 'rb') as fid:
            return pickle.load(fid)