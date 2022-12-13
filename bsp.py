from typing import Iterable
import numpy as np

from scipy import integrate
from scipy.special import beta, betainc



##############################################################################################################
##
##積分による重畳原理
##

##############################################################################################################


def bsp(start_t: float, end_t: float, time: Iterable(float),
        ind_dev:callable, property_func:callable):
    """
    積分範囲(start_t, end_t)
    property_func(time-g)*ind_dev(g)dg

    Parameters
    ----------
    start_t, end_t: float
        積分範囲
    time : Iterable(float)
        時間のリスト
    ind_dev : callable
        押し込みに関連する時間の関数
        df(δ(t))/dt
        ex ) f(δ) = δ**(3/2)
    property_func : callable
        押し込みに関連する時間の関数
        ex ) E(t) = Einf-(E0-Einf)t^-α

    Returns
    -------
    積分結果
        
    """
    def bsp_each_time(start_t: float, end_t: float, t_now: float,
                      ind_dev:callable,property_func:callable):
        def _integral_inner(g):
            return ind_dev(g)*property_func(t_now-g)
        return integrate.quad(_integral_inner, start_t, end_t)[0]
    return [  bsp_each_time(start_t, end_t, t_now, ind_dev, property_func) for t_now in time ]




##############################################################################################################
##
##β関数による重畳原理
##
##############################################################################################################

def _bsp_beta( t, k_exp, alpha, integral_range, t_dash = 1/50000):
    beta_a = k_exp+1
    beta_b = 1-alpha

    c_ = ((t+t_dash)**(k_exp-alpha+1))*(t_dash)**(alpha)
    beta_end = integral_range[1]/(t+t_dash)
    beta_start = integral_range[0]/(t+t_dash)
    beta_incomp = betainc(beta_a,beta_b,beta_end)-betainc(beta_a,beta_b,beta_start)
    c_beta = beta(beta_a, beta_b)*beta_incomp
    return c_*c_beta


##関数コールするようにしてまとめた方がいいかも
def _beta_einf_bsp_beta(k,integral_range):
    return (integral_range[1]**(k+1)-integral_range[0]**(k+1))/(k+1)

def bsp_beta_e0(time, integral_start, integral_end, 
                    e0, einf, alpha,
                    k_exp, k_coeff):
    def _bsp_beta_e0(t, e0, einf, alpha, k_exp,k_coeff,integral_range):
        bsp_res=_bsp_beta(t, k_exp, alpha,integral_range)
        return k_coeff*(einf*_beta_einf_bsp_beta(k_exp,integral_range)+(e0-einf)*bsp_res) 
    bsp_beta = np.sum([_bsp_beta_e0(time,
                               e0, einf, alpha,
                               k_exp_, k_coeff_,
                               integral_range=(integral_start,integral_end)
                               )
                    for k_exp_,k_coeff_ 
                    in zip(k_exp, k_coeff)],axis=0)
    return bsp_beta

def bsp_beta_e1(time, integral_start, integral_end, 
                    e1, einf, alpha,
                    k_exp, k_coeff):
    def _bsp_beta_e1(t, e1,einf,alpha,k_exp,k_coeff,integral_range, t_dash=1/50000):
        bsp_beta_res=_bsp_beta(t, k_exp, alpha,integral_range)
        return k_coeff*(einf*_beta_einf_bsp_beta(k_exp,integral_range)+(e1-einf)*((t_dash+t)**(1-alpha+k_coeff))*bsp_beta_res) 
    bsp_beta = np.sum([_bsp_beta_e1(time,
                               e1, einf, alpha,
                               k_exp_, k_coeff_,
                               integral_range=(integral_start,integral_end)
                               )
                    for k_exp_,k_coeff_ 
                    in zip(k_exp, k_coeff)],axis=0)
    return bsp_beta



##############################################################################################################
##
##応力緩和用ラッパー
##
##############################################################################################################
def bsp_sr_e0(t, 
                                 e0, einf, alpha,
                                 t_trig,
                                 k_exp_app,k_coeff_app,
                                 k_exp_srs=(0,0),k_coeff_srs=(0,0)):
    t_app = t[t_trig>=t]
    t_sr = t[t_trig<t]
    bsp_app,bsp_in_sr=[],[]

    if len(t_app)>0:
        bsp_app = bsp_beta_e0(t_app,0,t_app, e0, einf, alpha, k_exp_app, k_coeff_app)

    if len(t_sr)>0:
        bsp_app_in_sr = bsp_beta_e0(t_sr,0,t_app, e0, einf, alpha, k_exp_app, k_coeff_app)
        bsp_sr = bsp_beta_e0(t_sr,t_trig, t_sr, e0, einf, alpha, k_exp_srs, k_coeff_srs)
        bsp_sr+=bsp_app_in_sr
    bsp=np.append(bsp_app,bsp_in_sr)
    return bsp

def bsp_sr_e1(t, e1,einf,alpha,t_trig,
                                 k_exp_app,k_coeff_app,k_exp_srs=(0,0),k_coeff_srs=(0,0)):
    t_app = t[t_trig>=t]
    t_sr = t[t_trig<t]
    bsp_app,bsp_in_sr=[],[]
    if len(t_app)>0:
        bsp_app = bsp_beta_e1(t_app,0,t_app, e1, einf, alpha, k_exp_app, k_coeff_app)
    if len(t_sr)>0:
        bsp_app_in_sr = bsp_beta_e1(t_sr, 0, t_app, e1,einf,alpha,k_exp_app,k_coeff_app)
        bsp_sr = bsp_beta_e1(t_sr, t_trig, t_sr, e1, einf, alpha, k_exp_srs, k_coeff_srs)

        bsp_app_in_sr+=bsp_sr
    bsp=np.append(bsp_app,bsp_in_sr)
    return bsp