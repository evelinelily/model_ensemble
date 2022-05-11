import random
import numpy as np

def sample_range(low, high, size=None):
    assert high > low
    if size is None:
        return random.random() * (high-low) + low
    else:
        if isinstance(size, int):
            size = [size]
        return np.random.rand(*size) * (high - low) + low
