#%%
import numpy as np
from force_curve import ForceVolumeCurve


def get_contact_point(fc:ForceVolumeCurve):
    pass


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
                    xstep = config.loc["Xstep"],
                    ystep = config.loc["Ystep"])
# %%
