#%%
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass


import numpy as np
#%%

@dataclass
class ForceCurveAbstract(ABCMeta = ABCMeta):
    
    deflection:np.ndarray[object]
    zsensor:np.ndarray[object]
    
    xstep:int
    ystep:int
    
    cp:np.ndarray[int]=None

    force:np.ndarray[float]=None
    indentation:np.ndarray[float]=None    
    

    xstep_length:float = None
    ystep_length:float = None

    zig:bool = None

    def __post_init__(self):
        self.mapping_shape = (self.xstep, self.ystep)
        if self.zig:
            self.index = np.arange(len(self.deflection))
            self.index = self.index.reshape(self.mapping_shape)
            self.index[1::2,:] = self.index[1::2,::-1]
            self.index = self.index.reshape(-1,1)
