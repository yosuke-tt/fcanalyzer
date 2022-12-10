import numpy as np
from scipy import integrate


from properties import *

def boltz_mann_superposition_principle(t,
                                        delta_dev:callable,
                                        property_func:callable = et_e0,
                                        ):
    def _integral_inner(t):
        return delta_dev(t)*property_func(t)
    return integrate.quad(_integral_inner, 0, t)


