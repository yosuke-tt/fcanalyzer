#%%
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass


import numpy as np
from afm import AFM

#%%

@dataclass
class ForceCurveAbstract(metaclass = ABCMeta):
    
    afm:AFM
    
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
        self.zsensor*=self.afm.um_per_v  
        self.mapping_shape = (self.xstep, self.ystep)
        if self.zig:
            self.set_zig_idx()
        self.set_apppoint()
    def set_pre_ind_force(self):
        cantilever_deformation = self.deflection*self.afm.invols
        self.indentation_pre:np.ndarray[object] = self.zsensor-cantilever_deformation 
        self.force_pre:np.ndarray[object] = cantilever_deformation*self.afm.k

    def set_zig_idx(self):
        self.index:np.ndarray[int] = np.arange(len(self.deflection))
        self.index = self.index.reshape(self.mapping_shape)
        self.index[1::2,:] = self.index[1::2,::-1]
        self.index = self.index.reshape(-1,1)

    def set_apppoint(self):
        self.app_point_def = np.argmax(self.deflection, axis=1)
        self.app_point_zsen = np.argmax(self.zsensor, axis=1)
        self.diff_max_idx  = self.app_point_def - self.app_point_zsen 