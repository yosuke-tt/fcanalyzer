#%%
from typing import Iterable

import numpy as np
from scipy.optimize import minimize, brute, curve_fit
import matplotlib.pyplot as plt


from force_curve import ForceVolumeCurve






def _gen_ind_func(ind_app_func, ind_fit_ret, max_ind_time):
    def ind_func(time:Iterable[float]):
        time = np.array(time)
        time_app = time[time<=max_ind_time]
        time_ret = time[time>max_ind_time]
        return np.hstack([ind_app_func(time_app), ind_fit_ret(time_ret)])
    return ind_func

def _gen_ind_dev_func(ind_dev_app, ind_dev_ret_func, max_ind_time):
    def ind_dev_func(time:Iterable[float]):
        if isinstance(time,float):
            time = np.array([time])
        decline = np.zeros(len(time))
        time_app = time[time<=max_ind_time]
        time_ret = time[time>max_ind_time]
        decline[time<=max_ind_time]+=ind_dev_app(time_app)
        decline[time>max_ind_time] +=ind_dev_ret_func(time_ret)
        return decline
    return ind_dev_func

def _exp(time, *args):
    return np.sum([ a*time**b 
                   for a,b in zip(args[:len(args)//2],args[len(args)//2:])],
                   axis=0)

def _dev_coeff(args):
    dev_tmp =  np.hstack([ [a*b, b-1] 
                      for a,b in zip(args[:len(args)//2],args[len(args)//2:]) if b!=0])
    return np.hstack([dev_tmp[::2],dev_tmp[1::2]])

def _fix_multi_func(time, fix_point, *args):
    return np.sum([a*(time-fix_point[0])**b if b!=0  else fix_point[1] 
            for a, b in zip(args[:len(args)//2],args[len(args)//2:])],axis=0)



def gen_ind_2funcs_multi(time, indentation, fit_deg=10):

    max_ind_idx  = np.argmax(indentation)
    max_ind_time = time[max_ind_idx]
    ind_app_func_coeff = np.polyfit(time[:max_ind_idx], 
                                                indentation[:max_ind_idx],
                                                deg=fit_deg)
    ind_app_func  = np.poly1d(ind_app_func_coeff)
    max_ind_point =[ max_ind_time, ind_app_func(max_ind_time)]
    ind_ret_coeff = curve_fit(lambda time, *args:_fix_multi_func(time, 
                                                                           max_ind_point,
                                                                           *args,*np.arange(fit_deg+1)),
                                        time[max_ind_idx:],
                                        indentation[max_ind_idx:],
                                        p0 = -np.ones(fit_deg+1),maxfev=10000,
                                        ftol=2.23e-12, xtol=2.23e-12, gtol=2.23e-12
                                        )[0]
    ind_ret_coeff = np.hstack([ind_ret_coeff,np.arange(fit_deg+1)])
    ind_fit_ret = lambda time : _fix_multi_func(time, max_ind_point, *ind_ret_coeff)
    ind_fit_dev_ret_coeff=_dev_coeff(ind_ret_coeff)
    ind_dev_ret_func =lambda time: _fix_multi_func(time, [max_ind_point[0],ind_fit_dev_ret_coeff[::2][0]], 
                                                          *ind_fit_dev_ret_coeff)
    
    ind_func =     _gen_ind_func(ind_app_func, ind_fit_ret, max_ind_time)
    ind_dev_func = _gen_ind_dev_func(ind_app_func.deriv(),
                                                     ind_dev_ret_func,
                                                     max_ind_time)
    ind_32_dev_func = lambda t:(3/2)*ind_dev_func(t)*(ind_func(t)**(1/2))
        
    return ind_func, ind_dev_func, ind_32_dev_func, (max_ind_idx, max_ind_time)
                
    
    
def gen_ind_2funcs_exp(time, indentaion, num_dim=2):

    max_ind_idx  = np.argmax(indentaion)
    max_ind_time = time[max_ind_idx]
    
    p0 = np.hstack([np.ones(num_dim), np.arange(num_dim)])
    
    ind_app_coeff  = curve_fit(_exp,
                                time[:max_ind_idx], indentaion[:max_ind_idx],
                                p0 = p0,
                                maxfev=10000,ftol=2.23e-12, xtol=2.23e-12, gtol=2.23e-12
                                )[0]
    ind_dev_app_coeff = _dev_coeff(ind_app_coeff)

    ind_dev_app  = lambda time: _exp(time, *ind_dev_app_coeff)
    ind_app_func = lambda time: _exp(time, *ind_app_coeff)

    max_ind_point = [time[max_ind_idx], ind_app_func(time[max_ind_idx])]

    ind_ret_coeff = curve_fit(lambda time, *args:_fix_multi_func(time, max_ind_point, *args),
                                    time[max_ind_idx:], indentaion[max_ind_idx:],
                                    p0 = np.hstack([-np.ones(num_dim)*1e-5,np.arange(num_dim)/2]),
                                    bounds = (np.hstack([-np.ones(num_dim)*np.inf,np.zeros(num_dim)]),
                                              np.hstack([np.ones(num_dim),np.ones(num_dim)*np.inf])),
                                    maxfev=1000000,ftol=2.23e-12, xtol=2.23e-12, gtol=2.23e-12
                                    )[0]
    ind_fit_ret = lambda time : _fix_multi_func(time, max_ind_point, *ind_ret_coeff)
    
    ind_dev_ret_coeff = _dev_coeff(ind_ret_coeff)

    ind_dev_ret =lambda time: _fix_multi_func(time, [max_ind_point[0],ind_dev_ret_coeff[::2][0]], 
                                                          *ind_dev_ret_coeff)

    ind_func =     _gen_ind_func(ind_app_func, ind_fit_ret, max_ind_time)
    ind_dev_func = _gen_ind_dev_func(ind_dev_app,
                                                    ind_dev_ret,
                                                    max_ind_time)
    ind_32_dev_func = lambda t:(3/2)*ind_dev_func(t)*(ind_func(t)**(1/2))
    return ind_func, ind_dev_func, ind_32_dev_func, (max_ind_idx, max_ind_time)

def gen_ind_2funcs_multi_for_bb(time, indentaion, fit_deg=10):

    max_ind_idx  = np.argmax(indentaion)
    max_ind_time = time[max_ind_idx]
    
    ind_app_coeff = np.polyfit(time[:max_ind_idx], indentaion[:max_ind_idx], deg=fit_deg)
    ind_app_func  = np.poly1d(ind_app_coeff)

    ind_app_coeff = np.hstack([ind_app_coeff,np.arange(fit_deg,-1,-1)])
    ind_app_dev_coeff = ind_app_func.deriv()
    ind_app_dev_coeff = np.hstack([ind_app_dev_coeff, np.arange(fit_deg-1,-1,-1)])
    print(ind_app_dev_coeff)

    ind_ret_coeff = np.polyfit(time[max_ind_idx:], indentaion[max_ind_idx:], deg=fit_deg)
    ind_ret_func  = np.poly1d(ind_ret_coeff)
    ind_ret_coeff = np.hstack([ind_ret_coeff, np.arange(fit_deg,-1,-1)])

    ind_ret_dev_coeff = ind_ret_func.deriv().coefficients
    ind_ret_dev_coeff = np.hstack([ind_ret_dev_coeff,np.arange(fit_deg-1,-1,-1)])

    ind_pos_idx = indentaion[:max_ind_idx]>0
    ind32_app_coeff = np.polyfit(time[:max_ind_idx][ind_pos_idx],
                                 indentaion[:max_ind_idx][ind_pos_idx]**(3/2), deg=fit_deg)
    ind32_app_func  = np.poly1d(ind32_app_coeff)
    ind32_app_dev_coeff = ind32_app_func.deriv().coefficients
    ind32_app_dev_coeff = np.hstack([ind32_app_dev_coeff,np.arange(fit_deg-1,-1,-1)])
    return  (ind_app_coeff,ind_ret_coeff),\
            (ind_app_dev_coeff, ind_ret_dev_coeff),\
            ind32_app_dev_coeff,\
            (max_ind_idx, max_ind_time), ind_pos_idx


def gen_ind_2funcs_exp_for_bb(time, indentaion, num_dim=2):

    max_ind_idx  = np.argmax(indentaion)
    max_ind_time = time[max_ind_idx]
    
    p0 = np.hstack([np.ones(num_dim)*1e-6,np.arange(num_dim)])

    ind_app_coeff = curve_fit(lambda time, *args:_exp(time,*args),
                              time[:max_ind_idx], indentaion[:max_ind_idx],
                              p0 = p0, maxfev=10000,ftol=2.23e-12, xtol=2.23e-12, gtol=2.23e-12
                              )[0]
    ind_max = _exp(time[max_ind_idx],*ind_app_coeff)
    p0_ret = np.hstack([2*ind_max,-30e-3,-1*np.zeros(num_dim-2),
                    np.arange(num_dim)])

    ind_ret_coeff = curve_fit(lambda time, *args:_exp(time, *args),
                              time[max_ind_idx:], indentaion[max_ind_idx:],
                            p0 = p0_ret,
                            bounds = (np.hstack([-np.ones(num_dim)*np.inf, -np.zeros(num_dim)]),
                                      np.hstack([np.ones(num_dim)*np.inf,         np.ones(num_dim)*np.inf])),
                            maxfev=100000,ftol=2.23e-12, xtol=2.23e-12, gtol=2.23e-12
                              )[0]
    ind_app_dev_coeff = _dev_coeff(ind_app_coeff)
    ind_ret_dev_coeff = _dev_coeff(ind_ret_coeff)
    ind_pos_idx = indentaion[:max_ind_idx]>0
    ind32_app_coeff = curve_fit(lambda time, *args:_exp(time,*args),
                              time[:max_ind_idx][ind_pos_idx], indentaion[:max_ind_idx][ind_pos_idx]**(3/2),
                              p0 = p0, maxfev=10000,ftol=2.23e-12, xtol=2.23e-12, gtol=2.23e-12
                              )[0]
    ind32_app_dev_coeff = _dev_coeff(ind32_app_coeff)
    return  (ind_app_coeff,ind_ret_coeff), \
            (ind_app_dev_coeff, ind_ret_dev_coeff),\
            ind32_app_dev_coeff, \
            (max_ind_idx, max_ind_time), \
            ind_pos_idx


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


def set_baseline(force_w_offset, indenation_w_offset, app_bl_num=7000, ret_bl_num=5000):
    app_point = np.argmax(force_w_offset)
    app_x     = np.arange(app_point)
    ret_x     = np.arange(len(force_w_offset)-app_point)
    ret_x     +=len(app_x)-len(ret_x)

    force_app = force_w_offset[:app_point]
    force_ret = force_w_offset[app_point:][::-1]

    app_baseline = np.polyfit(app_x[:app_bl_num],force_w_offset[:app_point][:app_bl_num],deg=1)
    ret_baseline = np.polyfit(ret_x[:ret_bl_num],force_ret[:ret_bl_num],deg=1)
    med_baseline = np.median(np.vstack([app_baseline,ret_baseline]),axis=0)

    app_baseline_func=np.poly1d(app_baseline)
    ret_baseline_func=np.poly1d(ret_baseline)
    med_baseline_func=np.poly1d(med_baseline)

    app_baseline_func=np.poly1d(app_baseline-med_baseline)
    ret_baseline_func=np.poly1d(ret_baseline-med_baseline)


    force_app-=med_baseline_func(app_x)
    force_ret-=med_baseline_func(ret_x)


    force_app-=app_baseline_func(app_x)
    force_ret-=ret_baseline_func(ret_x)
    force_ret *=np.max(force_app)/np.max(force_ret)

    force = np.hstack([force_app, force_ret[::-1]])
    #ここで関数分ける。
    cp = np.where(np.abs(force_app)<0.01e-9)[0][-1]

    ret_cp = np.where(indenation_w_offset[app_point:]<indenation_w_offset[cp])[0][0]
    ret_cp+=app_point
    indenation=indenation_w_offset - indenation_w_offset[cp]
    return force[cp:ret_cp],indenation[cp:ret_cp], force, indenation