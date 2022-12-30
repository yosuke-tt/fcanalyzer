
#%%
import sys
sys.path.append(".")
import importlib
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


from scipy.optimize import  brentq


import bsp
importlib.reload(bsp)
from bsp import bsp, bsp_beta_e0, bsp_beta_e1
#%%







def ting_bata(time:Iterable[float],
              top_t:float,
              properties_params:Iterable[float], 
              ind_dev_app_exp:Iterable[float],
              ind_dev_app_coeff:Iterable[float],
              ind_dev_ret_exp:Iterable[float],
              ind_dev_ret_coeff:Iterable[float],
              ind_23_dev_func_exp:Iterable[float],
              ind_23_dev_func_coeff:Iterable[float],
              beta_e_func:str="e0"
              ):

    """β関数に変換したティングモデル（高速化のため）

    Parameters
    ----------
    time : Iterable[float]
        求める時間のリスト
    top_t : float
        dδ/dt==0の時間
    efunc_p : callable
        物性値の関数
    properties_params : Iterable[float]
        物性値の値
    ind_dev_app_coeff : Iterable[float]
        押し込み中の押し込み量フィッティングの係数
    ind_dev_app_exp : Iterable[float]
        押し込み中の押し込み量フィッティングの指数
        
    ind_dev_ret_coeff : Iterable[float]
        退避中の押し込み量フィッティングの係数
    ind_dev_ret_exp : Iterable[float]
        退避中の押し込み中の押し込み量フィッティングの指数

    indentation = ind_app_coeff*t**ind_app_exp


    ind_23_dev_func_coeff : Iterable[float]
        退避中の押し込み量の3/2乗のフィッティングの係数
    ind_23_dev_func_exp : Iterable[float]
        退避中の押し込み量の3/2乗のフィッティングの指数

    indentation**(3/2) = ind_23_dev_func_coeff*t**ind_23_dev_func_exp

    Returns
    -------
    _type_
        _description_
    """
    def _searching_t1_bata(time_ret, bsp_beta, properties):
        """t1をβ関数に変換した重畳で求める方法（高速化のため）

        Parameters
        ting_betaと同じ

        Returns
        -------
        _type_
            _description_
        """
        t1_pre=top_t
        t1s = np.zeros(len(time_ret))
        
        ret_beta = bsp_beta(time_ret, top_t, time_ret, *properties, ind_dev_app_exp, ind_dev_app_coeff)
        
        for i, t in enumerate(time_ret):
            ret_integrate = ret_beta[i]
            try:
                t1 = brentq(lambda t1:ret_integrate+bsp_beta(t, t1, top_t,*properties,
                                                                ind_dev_ret_exp,ind_dev_ret_coeff),0,t1_pre)
            except Exception as e:
                break
            t1s[i]+=t1
            t1_pre=t1
        return t1s
    bsp_beta_dict = {"e0":bsp_beta_e0,"e1":bsp_beta_e1}
    bsp_beta = bsp_beta_dict[beta_e_func]
    t1s = _searching_t1_bata(time[time>top_t],
                            bsp_beta, properties_params)
    app_int = bsp_beta(time[time<=top_t], 0, time[time<=top_t],*properties_params, 
                          ind_23_dev_func_exp, ind_23_dev_func_coeff )
    ret_int = bsp_beta(t1s, 0, t1s,*properties_params, 
                          ind_23_dev_func_exp, ind_23_dev_func_coeff )
    ting_curve=np.hstack([app_int,ret_int])
    return ting_curve




def ting(time:Iterable[float],
         properties_params:Iterable[float],
         top_t:float,
         indentation_dev_func:callable,
         indentation_23_dev_func:callable,
         efunc_p:callable):
    """ティングモデル

    Parameters
    ----------
    time : Iterable[float]
        _description_
    properties_params : Iterable[float]
        物性値
    top_t : float
        indentation_dev_funcの微分が変更する点
    indentation_dev_func : callable
        d(delta)/dt = indentation_dev_func(time)
    indentation_23_dev_func : callable
        d(delta**3/2)/dt = indentation_23_dev_func(time)
    efunc_p : callable
        ex)efunc_p(t)=Einf+(Einf-E0)**t^-alpha
    """
    def _searching_t1(time_ret, top_t,indentation_dev_func,efunc):
        """

        Parameters
        ----------
            ting関数と同じ
        Returns
        -------
        _type_
            _description_
        """
        t1_pre=top_t
        t1s = np.zeros(len(time_ret))
        #TODO:　βの方と同じにしてもいいかも
        ret_integrate = bsp(top_t, time_ret, time_ret, indentation_dev_func, efunc)

        for i, t in enumerate(time_ret):
            try:
                t1 = brentq(lambda t1:ret_integrate+bsp(t1,top_t,[t],indentation_dev_func,efunc),
                    0,t1_pre)
            except Exception as e:
                break
                
            t1s[i]+=t1
            t1_pre=t1
        return t1s

    efunc = lambda time:efunc_p(time,*properties_params)
    t1s = _searching_t1(time[time>top_t], top_t, indentation_dev_func, efunc)

    app_int = bsp(0, time[time<=top_t], time[time<=top_t], indentation_23_dev_func, efunc)
    ret_int = bsp(0, t1s,               t1s,               indentation_23_dev_func, efunc) 
    ting_curve=np.hstack([app_int,ret_int])
    return ting_curve


#%%

if __name__ =="__main__":
    pass