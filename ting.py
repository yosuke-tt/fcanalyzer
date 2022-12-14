
#%%
import numpy as np
import matplotlib.pyplot as plt


from scipy.optimize import  brentq

import sys
sys.path.append(".")
import importlib
import bsp
importlib.reload(bsp)
from bsp import bsp, bsp_beta_e0
#%%






import time as tt
def searching_t1_bata(time_ret, top_t, 
                      app_ind_dev_exp,app_ind_dev_coeff,
                      ret_ind_dev_exp,ret_ind_dev_coeff,
                      efunc, prop ):
    t1_pre=top_t
    t1s = np.zeros(len(time_ret))
    
    #TODO:Tを入れるだけの関数にして渡すようにする
    ret_beta = bsp_beta_e0(time_ret, top_t, time_ret, *prop, app_ind_dev_exp,app_ind_dev_coeff)
    
    for i, t in enumerate(time_ret):
        ret_integrate = ret_beta[i]
        try:
            # print(ret_integrate)
            t1 = brentq(lambda t1:ret_integrate+bsp_beta_e0(t, t1, top_t,*prop,
                                                            ret_ind_dev_exp,ret_ind_dev_coeff),0,t1_pre)
        except Exception as e:
            break
        t1s[i]+=t1
        t1_pre=t1
    return t1s


def ting_bata(time, 
              properties_params, 
              top_t,
              dec_app_coeff,dec_app_exp,
              dec_ret_coeff,dec_ret_exp,
              indentation_23_dev_func_coeff,
              indentation_23_dev_func_exp,
              efunc_p):
    
    efunc = lambda time:efunc_p(time,*properties_params)
    t1s = searching_t1_bata(time[time>top_t], top_t, 
                            dec_app_exp, dec_app_coeff, 
                            dec_ret_exp, dec_ret_coeff,
                            efunc, properties_params)
    #TODO: 時間の配列の関数に統一する。
    
    
    app_int = bsp_beta_e0(time[time<=top_t], 0, time[time<=top_t],*properties_params, 
                          indentation_23_dev_func_exp, indentation_23_dev_func_coeff )
    ret_int = bsp_beta_e0(t1s, 0, t1s,*properties_params, 
                          indentation_23_dev_func_exp, indentation_23_dev_func_coeff )
    ting_curve=np.hstack([app_int,ret_int])
    return ting_curve

def searching_t1(time_ret, top_t,indentation_dev_func,efunc):
    t1_pre=top_t
    t1s = np.zeros(len(time_ret))
    for i, t in enumerate(time_ret):
        #TODO:Tを入れるだけの関数にして渡すようにする
        ret_integrate = bsp(top_t, t, t, indentation_dev_func,efunc)
        try:
            t1 = brentq(lambda t1:ret_integrate+bsp(t1,top_t,t,indentation_dev_func,efunc),
                0,t1_pre)
        except Exception as e:
            break
            
        t1s[i]+=t1
        t1_pre=t1
    return t1s


def ting(time, properties_params,tp_idx, top_t,indentation_dev_func,indentation_23_dev_func,efunc_p):

    efunc = lambda time:efunc_p(time,*properties_params)
    t1s = searching_t1(time[tp_idx+1:], top_t, indentation_dev_func, efunc)
    #TODO: 時間の配列の関数に統一する。
    app_int = [bsp(0, t, t, indentation_23_dev_func, efunc) 
                for t in time[:tp_idx+1] ]
    ret_int = np.array([bsp(0, t, t,indentation_23_dev_func, efunc) 
                            for t in t1s ])
    ting_curve=np.hstack([app_int,ret_int])
    return ting_curve


#%%

if __name__ =="__main__":
    pass