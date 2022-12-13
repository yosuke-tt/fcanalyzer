#%%
import numpy as np
from scipy.optimize import minimize, brute, curve_fit

from force_curve import ForceVolumeCurve
def get_cp(deflection):
    return np.where(deflection[:len(deflection)//2]<0.005)[0][-1]


def gen_indentaion_func(time, indentaion, output_coeff = False, fit_dim=10):
    
    indentation_fit = np.polyfit(time, indentaion, deg=fit_dim)
    indentation_func = np.poly1d(indentation_fit)
    indentation_dev_func = indentation_func.deriv()
    indentation_23_dev_func = lambda t:3*indentation_dev_func(t)*(indentation_func(t)**(1/2))/2    
    if not output_coeff:
        return indentation_func, indentation_dev_func, indentation_23_dev_func
    else:
        return indentation_func, indentation_dev_func, indentation_23_dev_func

def preprocessing(deflection, 
                  zsensor, 
                  get_cp,
                  invols=30, 
                  um_per_v=25e-6, 
                  sampling_t=3e-6,
                  dim_fit=10):
    
    cp = get_cp(deflection)
    deflection-=deflection[cp]
    diff_top_idx = np.argmax(deflection) - np.argmax(zsensor)
    zsensor = zsensor[-diff_top_idx:]
    deflection = deflection[:diff_top_idx]
    ret_cp = np.argmax(zsensor)+np.where(zsensor[np.argmax(zsensor):]<zsensor[cp])[0][0]

    deformation_cantilever = deflection*invols*1e-9
    force      = deformation_cantilever*0.1

    indentation_pre = zsensor*um_per_v-deformation_cantilever
    
    diff_top_idx = np.argmax(indentation_pre)-np.argmax(force)
    force = force[cp:ret_cp]
    time = np.arange(0,len(force))*sampling_t

    indentation = indentation_pre[cp+diff_top_idx:ret_cp+diff_top_idx]-indentation_pre[cp+diff_top_idx]
    indentation_func, indentation_dev_func, indentation_23_dev_func = gen_indentaion_func(time,indentation)
    tp_idx = np.where(indentation_dev_func(time)>=0)[0][-1]
    diff_top_idx -= np.argmax(force)-tp_idx

    indentation = indentation_pre[cp+diff_top_idx:ret_cp+diff_top_idx]-indentation_pre[cp+diff_top_idx]
    indentation_pos_idx = indentation>0
    indentation_func, indentation_dev_func, indentation_23_dev_func = gen_indentaion_func(time[indentation_pos_idx], indentation[indentation_pos_idx])
    
    indentation_23_dev_func_for_beta = np.poly1d(np.polyfit(time[indentation_pos_idx], indentation[indentation_pos_idx]**(3/2),deg=dim_fit)).deriv()
    
    return force[indentation_pos_idx], indentation_func, indentation_dev_func, indentation_23_dev_func, indentation_23_dev_func_for_beta, time[indentation_pos_idx]

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
            

