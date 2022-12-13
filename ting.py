
#%%
import numpy as np
import matplotlib.pyplot as plt


from scipy.optimize import  brentq

import sys
sys.path.append(".")
import importlib
import boltzmanns_superposition_principle
importlib.reload(boltzmanns_superposition_principle)
from boltzmanns_superposition_principle import boltz_mann_superposition_principle, bsp_beta_for_e0
#%%


def get_cp(deflection):
    return np.where(deflection[:len(deflection)//2]<0.015)[0][-1]

def preprocessing(deflection, 
                  zsensor, 
                  get_cp,
                  invols=30, 
                  um_per_v=25e-6, 
                  sampling_t=3e-6):
    
    cp = get_cp(deflection)
    deflection-=deflection[cp]
    diff_top_idx = np.argmax(deflection) - np.argmax(zsensor)
    
    if diff_top_idx>0:
        zsensor = zsensor[:-diff_top_idx]
        deflection=deflection[diff_top_idx:]
    
    else:
        zsensor = zsensor[-diff_top_idx:]
        deflection=deflection[:diff_top_idx]

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
    
    indentation_23_dev_func_for_beta = np.poly1d(np.polyfit(time[indentation_pos_idx], indentation[indentation_pos_idx]**(3/2),deg=30)).deriv()
    
    return force[indentation_pos_idx], indentation_func, indentation_dev_func, indentation_23_dev_func, indentation_23_dev_func_for_beta, time[indentation_pos_idx]

def gen_indentaion_func(time, indentaion, output_coeff = False):
    
    indentation_fit = np.polyfit(time, indentaion, deg=30)
    indentation_func = np.poly1d(indentation_fit)
    indentation_dev_func = indentation_func.deriv()
    indentation_23_dev_func = lambda t:3*indentation_dev_func(t)*(indentation_func(t)**(1/2))/2    
    if not output_coeff:
        return indentation_func, indentation_dev_func, indentation_23_dev_func
    else:
        return indentation_func, indentation_dev_func, indentation_23_dev_func



import time as tt
def searching_t1_bata(time_ret, top_t,
                 indentation_dev_func,
                efunc,
                prop
                ):
    t1_pre=top_t
    t1s = np.zeros(len(time_ret))

    ind_dev_coeff = indentation_dev_func.coeffs
    ind_dev_exp =np.arange(indentation_dev_func.order+1)[::-1]

    ret_beta = bsp_beta_for_e0(time_ret, top_t, time_ret, *prop,  
                    ind_dev_exp, 
                    ind_dev_coeff)


    for i, t in enumerate(time_ret):
        ret_integrate = ret_beta[i]


        try:

            t1 = brentq(lambda t1:ret_integrate+bsp_beta_for_e0(t, t1, top_t,
                                                                   *prop,
                                                                   ind_dev_exp,
                                                                   ind_dev_coeff ),
                0,t1_pre, xtol=2.2e-23, rtol=8.881784197001252e-16)
        except Exception as e:
            break
            
        t1s[i]+=t1
        t1_pre=t1
    return t1s
def ting_bata(time, 
         properties_params,
         top_t,
         indentation_dev_func,
         indentation_23_dev_func,
         efunc_p):
    efunc = lambda time:efunc_p(time,*properties_params)
    t1s = searching_t1_bata(time[time>top_t],
                       top_t,
                        indentation_dev_func,
                        efunc,
                        properties_params)

    ind_23_dev_coeff = indentation_23_dev_func.coeffs
    ind_23_dev_exp =np.arange(indentation_23_dev_func.order+1)[::-1]

    app_int = bsp_beta_for_e0(time[time<top_t], 0, time[time<top_t],*properties_params,
                    ind_23_dev_exp, ind_23_dev_coeff )

    ret_int = bsp_beta_for_e0(t1s, 0, t1s,*properties_params,
                    ind_23_dev_exp,ind_23_dev_coeff )


    ting_curve=np.hstack([app_int,ret_int])
    return ting_curve


def searching_t1(time_ret, 
                 top_t,
                 indentation_dev_func,
                 efunc,
                ):
    t1_pre=top_t
    t1s = np.zeros(len(time_ret))
    for i, t in enumerate(time_ret):
        ret_integrate = boltz_mann_superposition_principle(top_t, t, t,
                                                            indentation_dev_func,
                                                            efunc
                                                            )

        try:
            t1 = brentq(lambda t1:ret_integrate+boltz_mann_superposition_principle(t1,top_t,t,
                                                                                   indentation_dev_func,
                                                                                   efunc),
                0,t1_pre)
        except Exception as e:
            print(e)
            break
            
        t1s[i]+=t1
        t1_pre=t1
    return t1s
def ting(time, properties_params,
         tp_idx, top_t,
         indentation_dev_func,
         indentation_23_dev_func,
         efunc_p):

    efunc = lambda time:efunc_p(time,*properties_params)
    
    t1s = searching_t1(time[tp_idx+1:], 
                       top_t,
                        indentation_dev_func,
                        efunc)

    app_int = [boltz_mann_superposition_principle(0, t, t,
                                                indentation_23_dev_func,
                                                efunc) 
                for t in time[:tp_idx+1] ]
    ret_int = np.array([boltz_mann_superposition_principle(0, t, t,
                                                           indentation_23_dev_func,
                                                           efunc) 
            for t in t1s ])
    ting_curve=np.hstack([app_int,ret_int])
    return ting_curve


#%%

if __name__ =="__main__":
    pass