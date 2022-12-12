#%%
import numpy as np

import matplotlib.pyplot as plt

from force_curve_abc import ForceCurveAbstract
from dataclasses import dataclass

class ForceVolumeCurve(ForceCurveAbstract):
    def __init__(self,afm,deflection,zsensor,xstep,ystep,xstep_length = 3e-6,ystep_length = 3e-6) -> None:
        return super().__init__(afm,deflection,zsensor,xstep,ystep,xstep_length = 3e-6,ystep_length = 3e-6)
#%%
if __name__=="__main__":

    from glob import glob
    import matplotlib.pyplot as plt
    import pandas as pd
    from afm import AFM
    from force_curve import ForceVolumeCurve
    fc = glob("D:/tsuboyama/AFM3/data_20221203/data_162649/ForceCurve/*")
    config = pd.read_csv("D:/tsuboyama/AFM3/data_20221203/data_162649/config.txt",
                        encoding="shift-jis",
                        index_col=0)

    fc = np.array([np.loadtxt(fcc) for fcc in fc])

    afm = AFM(5e-6, 200, 0.1, 2)
    
    fc = ForceVolumeCurve(afm =afm,
                    deflection = fc[:,:len(fc)//2],
                    zsensor = fc[:,len(fc)//2:],
                    xstep = config["Xstep"],
>>>>>>> fc_dataclass
                    ystep = config["Ystep"])