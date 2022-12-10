import numpy as np


from afm import AFM


def hertz_constant(afm:AFM):
    poission = 0.5
    return 4*np.sqrt(afm.radius)/(3*(1-poission**2))

def hertz_apperant(fc, hertz_young):
    return hertz_constant(fc.afm)*hertz_young*fc.indentation**(3/2)

def et_e0(t, e0, einf, alpha, constant_shape):
    return constant_shape*(e0-einf)*(1+t)**(-alpha)

def et_e1(t, e1, einf, alpha, constant_shape):
    return constant_shape*(e1-einf)*(t)**(-alpha)
