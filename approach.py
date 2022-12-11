#%%
import numpy as np
from scipy.optimize import minimize, brute, curve_fit

from force_curve import ForceVolumeCurve


def get_cp_under_th(deflection, 
                    baseline_num:int|float = 0.9,
                    def_th :float = None):
    
    if isinstance(baseline_num, float):
        baseline_num = int(np.argmax(deflection)*0.9)

    coeff_baseline_1d = np.polyfit(np.arange(baseline_num), 
                                    deflection[:baseline_num],
                                        deg=1)

    baseline_func = np.poly1d(coeff_baseline_1d) 
    deflection_based = deflection - baseline_func(np.arange(len(deflection)))
    def_th = -np.mean(np.abs(deflection_based[:baseline_num//2]))/2
    cp = np.where(deflection_based[:np.argmax(deflection_based)]<def_th)[0][-1]

    return cp, coeff_baseline_1d 

def set_contact_point(fc:ForceVolumeCurve,
                        cp_fc_type:str="u_th",
                        cp_func_args:list|tuple=(),
                        cp_func_kargs:dict={}):
    cp_func_dict = {"u_th":get_cp_under_th}
    cp_func = cp_func_dict[cp_fc_type]

    cps = np.zeros(len(fc.indentation))
    baseline_coeffs = np.zeros(len(fc.indentation),2)

    for i ,deflection in enumerate(fc.deflection):
        cp, baseline_coeff = cp_func(deflection[:len(deflection)//2],
                                 *cp_func_args,
                                    **cp_func_kargs)
        cps[i]+=cp
        baseline_coeffs[i]+=baseline_coeff
    return cps, baseline_coeffs
#%%
if __name__=="__main__":

    from glob import glob
    import matplotlib.pyplot as plt
    import pandas as pd
    from afm import AFM
    from force_curve import ForceVolumeCurve
    fc = glob("D:/tsuboyama/AFM3/data_20221203//data_163902/ForceCurve/*")
    config = pd.read_csv("D:/tsuboyama/AFM3/data_20221203/data_163902/config.txt",
                        encoding="shift-jis",
                        index_col=0)

    fc = np.array([np.loadtxt(fcc) for fcc in fc[:2]])


    afm = AFM(5e-6, 200, 0.1, 2)
    
    fc = ForceVolumeCurve(afm =afm,
                    deflection = fc[:,:len(fc[0])//2],
                    zsensor = fc[:,len(fc[0])//2:],
                    xstep = config.loc["Xstep"],
                    ystep = config.loc["Ystep"])
    cps, baseline_coeffs = set_contact_point(fc)
            

# %%
