#%%
def indentation(t):
    if t>1000*3e-9:
        return 6e-6-t
    else :
        return t
def indentation_dev(t):
    if t>1000*3e-9:
        return -1
    else :
        return 1
    
def indentatation_23_dev(t):
    return (3/2)*indentation_dev(t)*np.sqrt(indentation(t))


import numpy as np
import matplotlib.pyplot as plt
time = np.arange(2000)*3e-9
plt.plot(time, [indentation(t) for t in time])

def indentation_32(t):
    if t>1000*3e-9:
        return (6e-6-t)**(3/2)
    else :
        return t**(3/2)