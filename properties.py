from afm import AFM


def hertz_constant(afm:AFM):
    poission = 0.5
    return 4*np.sqrt(afm.radius)/(3*(1-poission**2))

def hertz_apperant(fc, hertz_young):
    return hertz_constant(fc.afm)*hertz_young*fc.indentation**(3/2)

