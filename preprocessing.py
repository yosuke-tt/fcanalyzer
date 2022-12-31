#%%
from typing import Iterable

import numpy as np
from scipy.optimize import minimize, brute, curve_fit
import matplotlib.pyplot as plt


from force_curve import ForceVolumeCurve






def _gen_indentation_func(indentation_app_func, indentation_fit_ret, max_ind_time):
    def indentation_func(time:Iterable[float]):
        time_app = time[time<=max_ind_time]
        time_ret = time[time>max_ind_time]
        return np.hstack([indentation_app_func(time_app), indentation_fit_ret(time_ret)])
    return indentation_func

def _gen_indentation_dev_func(indentation_dev_app, indentation_dev_fit_ret, max_ind_time):
    def indentation_dev_func(time:Iterable[float]):
        decline = np.zeros(len(time))
        time_app = time[time<=max_ind_time]
        time_ret = time[time>max_ind_time]
        
        decline[time<=max_ind_time]+=indentation_dev_app(time_app)
        decline[time>max_ind_time] +=indentation_dev_fit_ret(time_ret)
        return decline
    return indentation_dev_func

def _exp(time, *args):
    return np.sum([ a*time**b 
                   for a,b in zip(args[:len(args)//2],args[len(args)//2:])],
                   axis=0)

def _dev_coeff(*args):
    dev_tmp =  np.hstack([ [a*b, b-1] 
                      for a,b in zip(args[:len(args)//2],args[len(args)//2:]) if b!=0])
    return np.hstack([dev_tmp[::2],dev_tmp[1::2]])

def _fix_multi_func(time, fix_point, *args):
    return np.sum([a*(time-fix_point[0])**b if b!=0  else fix_point[1] 
            for a, b in zip(args[:len(args)//2],args[len(args)//2:])],axis=0)



def gen_indentaion_func_multi(time, indentaion, fit_deg=10, return_coeff=False):

    max_ind_idx  = np.argmax(indentaion)
    max_ind_time = time[max_ind_idx]
    indentation_app_func_coeff = np.polyfit(time[:max_ind_idx], 
                                                indentaion[:max_ind_idx],
                                                deg=fit_deg)
    indentation_app_func  = np.poly1d(indentation_app_func_coeff)
    max_ind_point =[ max_ind_time, indentation_app_func(max_ind_time)]
    indentation_ret_coeff = curve_fit(lambda time, *args:_fix_multi_func(time, 
                                                                           max_ind_point,
                                                                           *args,*np.arange(fit_deg+1)),
                                        time[max_ind_idx:],
                                        indentaion[max_ind_idx:],
                                        p0 = -np.ones(fit_deg+1),maxfev=10000,
                                        ftol=2.23e-12, xtol=2.23e-12, gtol=2.23e-12
                                        )[0]
    indentation_fit_ret = lambda time : _fix_multi_func(time, max_ind_point, *indentation_ret_coeff,*np.arange(fit_deg+1))
    indentation_fit_dev_ret_coeff=_dev_coeff(*indentation_ret_coeff,*np.arange(fit_deg+1))
    indentation_dev_fit_ret =lambda time: _fix_multi_func(time, [max_ind_point[0],indentation_fit_dev_ret_coeff[::2][0]], 
                                                          *indentation_fit_dev_ret_coeff[::2],
                                                          *indentation_fit_dev_ret_coeff[1::2])
    
    indentation_func =     _gen_indentation_func(indentation_app_func, indentation_fit_ret, max_ind_time)
    indentation_dev_func = _gen_indentation_dev_func(indentation_app_func.deriv(),
                                                    indentation_dev_fit_ret,
                                                    max_ind_time)
    indentation_32_dev_func = lambda t:(3/2)*indentation_dev_func(t)*(indentation_func(t)**(1/2))
    if not return_coeff:
        
        return indentation_func, indentation_dev_func, indentation_32_dev_func, max_ind_idx, max_ind_time
    else:
        ind_app_pos_idx = indentaion[:max_ind_idx]>0
        ind32_app_coeff         =np.polyfit(time[:max_ind_idx][ind_app_pos_idx], 
                                            indentaion[:max_ind_idx][ind_app_pos_idx]**(3/2),
                                            deg=fit_deg)
        ind32_app_dev_coeff = np.poly1d(ind32_app_coeff).deriv()
        ind32_app_dev_func =  np.poly1d(ind32_app_dev_coeff)

        return (indentation_func,
                (
                 np.hstack([indentation_app_func_coeff, np.arange(fit_deg,-1, -1)]),
                 np.hstack([indentation_ret_coeff, np.arange(fit_deg+1)])
                ),\
                (indentation_dev_func,
                 np.hstack([indentation_app_func.deriv().coefficients, np.arange(fit_deg,-1, -1)]),
                 np.hstack([indentation_ret_coeff, np.arange(fit_deg+1)])
                 ), \
                (ind32_app_dev_func,
                 np.hstack([ind32_app_dev_coeff, np.arange(fit_deg-1,-1, -1)])
                 ),\
                max_ind_idx, max_ind_time, ind_app_pos_idx
                )
    
    
def gen_indentaion_func_exp(time, indentaion, num_dim=2, return_coeff=False):

    max_ind_idx  = np.argmax(indentaion)
    max_ind_time = time[max_ind_idx]
    
    p0 = np.hstack([np.ones(num_dim), np.arange(num_dim)])
    
    indentation_app_coeff  = curve_fit(_exp,
                                        time[:max_ind_idx], indentaion[:max_ind_idx],
                                        p0 = p0,
                                        maxfev=10000,ftol=2.23e-12, xtol=2.23e-12, gtol=2.23e-12
                                        )[0]
    indentation_dev_app_coeff = _dev_coeff(*indentation_app_coeff)

    indentation_dev_app =lambda time: _exp(time, *indentation_dev_app_coeff[::2],
                                                 *indentation_dev_app_coeff[1::2])
    indentation_app_func = lambda time: _exp(time, *indentation_app_coeff)

    max_ind_point = [time[max_ind_idx], indentation_app_func(time[max_ind_idx])]

    indentation_ret_coeff = curve_fit(lambda time, *args:_fix_multi_func(time, 
                                                                        max_ind_point,
                                                                        *args),
                                    time[max_ind_idx:], indentaion[max_ind_idx:],
                                    p0 = np.hstack([-np.ones(num_dim)*1e-5,np.arange(num_dim)/2]),
                                    bounds = (np.hstack([-np.ones(num_dim)*np.inf,np.zeros(num_dim)]),
                                              np.hstack([np.ones(num_dim),np.ones(num_dim)*np.inf])),
                                    maxfev=1000000,ftol=2.23e-12, xtol=2.23e-12, gtol=2.23e-12
                                    )[0]
    indentation_fit_ret = lambda time : _fix_multi_func(time, max_ind_point, *indentation_ret_coeff)
    
    indentation_dev_ret_coeff = _dev_coeff(*indentation_ret_coeff)

    indentation_dev_ret =lambda time: _fix_multi_func(time, [max_ind_point[0],indentation_dev_ret_coeff[::2][0]], 
                                                          *indentation_dev_ret_coeff[::2],
                                                          *indentation_dev_ret_coeff[1::2])

    indentation_func =     _gen_indentation_func(indentation_app_func, indentation_fit_ret, max_ind_time)
    indentation_dev_func = _gen_indentation_dev_func(indentation_dev_app,
                                                    indentation_dev_ret,
                                                    max_ind_time)
    indentation_23_dev_func = lambda t:(3/2)*indentation_dev_func(t)*(indentation_func(t)**(1/2))
    if not return_coeff:
        return indentation_func, indentation_dev_func, indentation_23_dev_func, max_ind_idx, max_ind_time
    else:
        ind_app_pos_idx = indentaion[:max_ind_idx]>0
        ind32_app_coeff =curve_fit( _exp,
                                    time[:max_ind_idx][ind_app_pos_idx], 
                                    indentaion[:max_ind_idx][ind_app_pos_idx]**(3/2),
                                    p0=p0)[0]
        
        ind32_app_dev_coeff = _dev_coeff(*ind32_app_coeff)
        ind32_app_dev_func = lambda time:_exp(time,*ind32_app_coeff)
        return (indentation_func, 
                (indentation_app_coeff, indentation_ret_coeff),
                ),\
                (indentation_dev_func, 
                 (indentation_dev_app_coeff, indentation_dev_ret_coeff)
                 ), \
                (ind32_app_dev_func,
                 (ind32_app_dev_coeff[::2],ind32_app_dev_coeff[::2])
                 ), \
                max_ind_idx, max_ind_time, ind_app_pos_idx
                
                
def gen_indentaion_23_func_line_beta(time, indentaion):

    max_ind_idx = np.argmax(indentaion)
    app_23_func = lambda x, a, b, c: a*x**(3/2)+b*x+c
    indentation_app_func  = curve_fit(app_23_func,time[:max_ind_idx], indentaion[:max_ind_idx])[0]
    indentation_23_dev_func = lambda time : app_23_func(time, *indentation_app_func)

    return  indentation_23_dev_func, indentation_app_func,[3/2,1,0] 

def get_cp(deflection, def_th = 0.008):
    return np.where(deflection[:len(deflection)//2]<def_th)[0][-1]
def get_ret_cp(zsensor, cp):
    return np.argmax(zsensor)+np.where(zsensor[np.argmax(zsensor):]<zsensor[cp])[0][0]


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
