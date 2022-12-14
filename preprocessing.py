#%%
from typing import Iterable

import numpy as np
from scipy.optimize import minimize, brute, curve_fit
import matplotlib.pyplot as plt


from force_curve import ForceVolumeCurve





def gen_indentaion_func(time, indentaion, output_coeff = False, fit_dim=10):
    
    indentation_fit = np.polyfit(time, indentaion, deg=fit_dim)
    indentation_func = np.poly1d(indentation_fit)
    indentation_dev_func = indentation_func.deriv()
    indentation_23_dev_func = lambda t:3*indentation_dev_func(t)*(indentation_func(t)**(1/2))/2    
    return indentation_func, indentation_dev_func, indentation_23_dev_func


def gen_indentaion_func_line(time, indentaion):
    max_ind_idx = np.argmax(indentaion)
    max_ind_time = time[np.argmax(indentaion)]
    indentation_fit_app  = np.poly1d(np.polyfit(time[:max_ind_idx], 
                                      indentaion[:max_ind_idx],
                                      deg=1))
    indentation_fit_ret_ = curve_fit(lambda x, a: a*(x-time[max_ind_idx])+ indentaion[max_ind_idx], 
                                     time[max_ind_idx:],
                                     indentaion[max_ind_idx:])[0]
    indentation_fit_ret = lambda time:indentation_fit_ret_*(time-max_ind_time)+ indentaion[max_ind_idx]
    
    def gen_indentation_func(indentation_fit_app, indentation_fit_ret, max_ind_time):
        def indentation_func_line(time:Iterable[float]):
            time_app = time[time<=max_ind_time]
            time_ret = time[time>max_ind_time]
            return np.hstack([indentation_fit_app(time_app), indentation_fit_ret(time_ret)])
        return indentation_func_line
    
    def gen_indentation_dev_func(dec_app, dec_ret, max_ind_time):
        def indentation_dev_func_line(time:Iterable[float]):
            decline = np.zeros(len(time))
            decline[time<=max_ind_time]+=dec_app
            decline[time>max_ind_time]+=dec_ret    
            return decline
        return indentation_dev_func_line
    indentation_func = gen_indentation_func(indentation_fit_app, indentation_fit_ret, max_ind_time)
    indentation_dev_func = gen_indentation_dev_func(indentation_fit_app.coef[0],
                                                    indentation_fit_ret_,
                                                    max_ind_time)
    indentation_23_dev_func = lambda t:3*indentation_dev_func(t)*(indentation_func(t)**(1/2))/2    
    return indentation_func, indentation_dev_func, indentation_23_dev_func, indentation_fit_app.coef[0], indentation_fit_ret_

def gen_indentaion_23_func_line_beta(time, indentaion):

    max_ind_idx = np.argmax(indentaion)
    app_23_func = lambda x, a, b, c: a*x**(3/2)+b*x+c
    indentation_fit_app  = curve_fit(app_23_func,time[:max_ind_idx], indentaion[:max_ind_idx])[0]
    indentation_23_dev_func = lambda time : app_23_func(time, *indentation_fit_app)

    return  indentation_23_dev_func, indentation_fit_app,[3/2,1,0] 

def get_cp(deflection, def_th = 0.008):
    return np.where(deflection[:len(deflection)//2]<def_th)[0][-1]
def get_ret_cp(zsensor, cp):
    return np.argmax(zsensor)+np.where(zsensor[np.argmax(zsensor):]<zsensor[cp])[0][0]



def preprocessing(deflection, 
                  zsensor, 
                  get_cp,
                  invols=200, 
                  um_per_v=25e-6, 
                  sampling_t=10e-6,
                  dim_fit=10,
                  is_line_ind = True):
    
    cp = get_cp(deflection)
    deflection-=deflection[cp]
    diff_top_idx = np.argmax(deflection) - np.argmax(zsensor)
    zsensor = zsensor[-diff_top_idx:]
    deflection = deflection[:diff_top_idx]
    ret_cp = get_ret_cp(zsensor, cp)
    
    deformation_cantilever = deflection*invols*1e-9
    force      = deformation_cantilever*0.1
    indentation_pre = zsensor*um_per_v-deformation_cantilever
    diff_top_idx = np.argmax(indentation_pre)-np.argmax(force)
    force = force[cp:ret_cp]
    time = np.arange(0,len(force))*sampling_t
    
    indentation = indentation_pre[cp+diff_top_idx:ret_cp+diff_top_idx]-indentation_pre[cp+diff_top_idx]
    
    
    if is_line_ind:
        indentation_func, indentation_dev_func, indentation_23_dev_func, dec_app, dec_ret = gen_indentaion_func_line(time,indentation)
    else:
        indentation_func, indentation_dev_func, indentation_23_dev_func = gen_indentaion_func(time,indentation)
    tp_idx = np.where(indentation_dev_func(time)>=0)[0][-1]
    diff_top_idx -= np.argmax(force)-tp_idx

    indentation = indentation_pre[cp+diff_top_idx:ret_cp+diff_top_idx]-indentation_pre[cp+diff_top_idx]
    indentation_pos_idx = indentation>0
    if is_line_ind:
        indentation_func, indentation_dev_func, indentation_23_dev_func, dec_app, dec_ret = gen_indentaion_func_line(time[indentation_pos_idx], indentation[indentation_pos_idx])
    else:
        indentation_func, indentation_dev_func, indentation_23_dev_func = gen_indentaion_func(time[indentation_pos_idx], indentation[indentation_pos_idx])
    
    if is_line_ind:
        indentation_23_dev_func_for_beta = gen_indentaion_23_func_line_beta(time, indentation[indentation_pos_idx])
    else:
        indentation_23_dev_func_for_beta = np.poly1d(np.polyfit(time[indentation_pos_idx], indentation[indentation_pos_idx]**(3/2),deg=dim_fit)).deriv()

    return force[indentation_pos_idx], indentation_func, indentation_dev_func, indentation_23_dev_func, indentation_23_dev_func_for_beta, time[indentation_pos_idx], dec_app, dec_ret

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
            

