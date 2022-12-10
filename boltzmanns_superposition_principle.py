import numpy as np
from scipy import integrate


from properties import *

def boltz_mann_superposition_principle(t,
                                        delta_dev:callable,
                                        property_func:callable = et_e0,
                                        ):
    def _integral_inner(t):
        return delta_dev(t)*property_func(t)
    return integrate.quad(_integral_inner, 0, t)


warnings.simplefilter('ignore') 
def beta_bsp( t, k_exp, alpha,integral_range, t_dash = 1/50000):
    beta_a = k_exp+1
    beta_b = 1-alpha
    c_ = ((t+t_dash)**(k_exp-alpha+1))*(t_dash)**(alpha)
    beta_end = integral_range[1]/(t+t_dash)
    beta_start = integral_range[0]/(t+t_dash)
    beta_incomp = stats.beta.cdf(beta_end,beta_a,beta_b)-stats.beta.cdf(beta_start,beta_a,beta_b)
    c_beta = beta(beta_a, beta_b)*beta_incomp
    return c_*c_beta, c_

def beta_einf(k,integral_range):
    return (integral_range[1]**(k+1)-integral_range[0]**(k+1))/(k+1)

def _bsp(t, e0,einf,alpha,k_exp,k_coeff,integral_range):
    beta_bsp_=beta_bsp(t, k_exp, alpha,integral_range)
    return k_coeff*(einf*beta_einf(k_exp,integral_range)+(e0-einf)*beta_bsp_[0]) 

def e1_bsp(t, e1,einf,alpha,k_exp,k_coeff,integral_range, t_dash=1/50000):
    beta_bsp_=beta_bsp(t, k_exp, alpha,integral_range)
    return k_coeff*(einf*beta_einf(k_exp,integral_range)+(e1-einf)*((t_dash+t)**(1-alpha+k_coeff))*beta_bsp_[0]) 


def bsp_stress_relaxation_for_e0(t, e0,einf,alpha,t_trig,k_exp_app,k_coeff_app,k_exp_srs=(0,0),k_coeff_srs=(0,0)):
    t_app = t[t_trig>=t]
    t_sr = t[t_trig<t]
    bsp_app,bsp_in_sr=[],[]

    if len(t_app)>0:
        bsp_app = _bsp(t_app, e0,einf,alpha,k_exp_app,k_coeff_app,integral_range=(0,t_app))
    if len(t_sr)>0:
        bsp_app_in_sr = _bsp(t_sr, e0,einf,alpha,k_exp_app,k_coeff_app,integral_range=(0,t_trig))
        bsp_sr = np.sum([_bsp(t_sr, e0,einf,alpha,k_exp_sr, k_coeff_sr, integral_range=(t_trig,t_sr)) \
                    for k_exp_sr,k_coeff_sr in zip(k_exp_srs,k_coeff_srs)],axis=0)
        bsp_in_sr=bsp_app_in_sr+bsp_sr
    bsp=np.append(bsp_app,bsp_in_sr)
    return hertz_constant*bsp

def bsp_stress_relaxation_for_e1(t, e1,einf,alpha,t_trig,k_exp_app,k_coeff_app,k_exp_srs=(0,0),k_coeff_srs=(0,0)):
    t_app = t[t_trig>=t]
    t_sr = t[t_trig<t]
    bsp_app,bsp_in_sr=[],[]
    if len(t_app)>0:
        bsp_app = e1_bsp(t_app, e1,einf,alpha,k_exp_app,k_coeff_app,integral_range=(0,t_app))
    if len(t_sr)>0:
        bsp_app_in_sr = e1_bsp(t_sr, e1,einf,alpha,k_exp_app,k_coeff_app,integral_range=(0,t_trig))
        bsp_sr = np.sum([e1_bsp(t_sr, e1,einf,alpha,k_exp_sr, k_coeff_sr, integral_range=(t_trig,t_sr)) \
                    for k_exp_sr,k_coeff_sr in zip(k_exp_srs,k_coeff_srs)],axis=0)
        bsp_in_sr=bsp_app_in_sr+bsp_sr
    bsp=np.append(bsp_app,bsp_in_sr)
    return hertz_constant*bsp
