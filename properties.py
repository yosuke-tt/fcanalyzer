
import numpy as np


from afm import AFM


T_DASH=1/50000

def hertz_constant(afm:AFM):
    poission = 0.5
    return 4*np.sqrt(afm.radius)/(3*(1-poission**2))

def hertz_apperant(fc, hertz_young):
    return hertz_constant(fc.afm)*hertz_young*fc.indentation**(3/2)

def et_e0(t, e0, einf, alpha):
    return einf+(e0-einf)*(1+t/T_DASH)**(-alpha)

def et_e1(t, e1, einf, alpha):
    return einf+(e1-einf)*(t+T_DASH)**(-alpha)
    
def tension(fc, tension):
    return np.pi*tension*fc.indentation

