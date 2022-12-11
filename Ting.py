
import numpy as np
import matplotlib.pyplot as plt


from scipy.optimize import curve_fit, brenth
from scipy import integrate  
from scipy.optimize import newton



from properties import hertz_constant
from afm import AFM 
from properties import et_e0
from boltzmanns_superposition_principle import boltz_mann_superposition_principle

#%%


def searching_t1(time,
                t_turning_point,
                turning_point,
                indentation_dev_func,
                efunc):

    t1_results = []
    t1_pre = t_turning_point
    def _app_integrate_t1(t1):
        return boltz_mann_superposition_principle(t_turning_point, indentation_dev_func, efunc, start_t=t1)
    for t in time[turning_point:]:
        ret_integrate = boltz_mann_superposition_principle(t, indentation_dev_func, efunc, start_t=t_turning_point) 
        try:
            t1 = brenth(lambda t1:ret_integrate+_app_integrate_t1(t1),
                    0,t_turning_point, xtol=2.2e-23, rtol=8.881784197001252e-16)
        except Exception as e:
            t1 = 0
        t1_results.append(t1)
        t1_pre=t1

    return t1_results
def bsp_approach(time, e0, einf, alpha):
    efunc = lambda t:hc*et_e0(t,e0=e0, einf=einf, alpha=alpha)
    f = np.array([boltz_mann_superposition_principle(t, indentation_23_dev_func,efunc) for t in time])
    return f*1e9

def ting(time, e0, einf, alpha, 
        t_turning_point,
        indentation_23_dev_func, 
        indentation_dev_func):
    efunc = lambda t: hc*et_e0(t, e0=e0, einf=einf, alpha=alpha)
    turning_point = np.where(time<t_turning_point)[0][-1]

    force_app = [boltz_mann_superposition_principle(t, indentation_23_dev_func,efunc) 
                    for t in time[:turning_point]]
                    
    t1 = searching_t1(time, t_turning_point,turning_point, indentation_dev_func, efunc)
    force_ret = [boltz_mann_superposition_principle(t, indentation_23_dev_func,efunc) 
                    for t in t1]
    return np.hstack([force_app, force_ret])

#%%
if __name__ == "__main__":
    fc = np.loadtxt("D:/tsuboyama/AFM3/data_20221204/data_192156/ForceCurve/ForceCurve_025.lvm")
    deflection = fc[:len(fc)//2]
    zsensor = fc[len(fc)//2:]


    app_point_sp_z = np.argmax(zsensor)
    app_point_sp_def = np.argmax(deflection)
    diff_app_point = app_point_sp_z-app_point_sp_def

    print(app_point_sp_z,app_point_sp_def,diff_app_point)
    app_point = 5000
    ret_point = 3000

    x_fit  = np.hstack([
                np.arange(app_point),
                np.arange(len(deflection)-ret_point,len(deflection))
            ])
    y_fit  = np.hstack([
                deflection[:app_point],
                deflection[-ret_point:]
            ])

    deflection_func = np.poly1d(np.polyfit(x_fit,y_fit, deg=2))
    deflection_base = deflection - deflection_func(np.arange(len(deflection))) 

    ret_cp = np.where(zsensor[app_point_sp_z:]<zsensor[diff_app_point+5700])[0][0]


    force_base = deflection_base[5700:app_point_sp_def+ret_cp]-deflection_base[5700]
    force      = force_base*300*0.1*1e-9
    # plt.plot(force)

    indentation = (zsensor[diff_app_point+5700:app_point_sp_z+ret_cp]-zsensor[diff_app_point+5700])*25e-6-force_base*300*1e-9
    ind_positive = indentation>0
    indentation = indentation[ind_positive]
    force = force[ind_positive]


    #%%

    indentation = np.arange(1000)*1e-9
    indentation = np.hstack([indentation,indentation[::-1]])
    plt.plot(indentation)
    #%%
    time = np.arange(0,len(indentation))*3e-6



    indentation_23_fit = np.polyfit(time, indentation**(3/2), deg=3)
    indentation_23_func = np.poly1d(indentation_23_fit)
    indentation_23_dev_func = indentation_23_func.deriv()

    from scipy.interpolate import UnivariateSpline
    spl = UnivariateSpline(time, indentation, k=3)

    indentation_fit = np.polyfit(time, indentation, deg=3)
    indentation_func = np.poly1d(indentation_fit)
    indentation_dev_func = indentation_func.deriv()



    plt.plot(time, indentation)
    plt.plot(time, indentation_func(time))
    # plt.plot(time, spl(time))
    # plt.show()
    # indentation_func = spl.derivative()
    # indentation_dev_func = spl.derivative()
    # plt.plot(time, spl.derivative(2)(time))
    plt.show()
    plt.close()
    plt.plot(time, indentation_dev_func.deriv()(time))
    #%%
    afm = AFM(radius=5e-6,invols=300, k=0.1,no_afm=3)
    hc = hertz_constant(afm)


    truning_point = np.where(indentation_dev_func(time)<0)[0][0]

    p = curve_fit(fit_func,time[:truning_point],force[:truning_point]*1e9,
                    method="trf", p0=(1000,0,0.2), bounds=((0,0,0),(100000,100,1)),
                    gtol=2.2e-16)
    plt.plot(time[:truning_point],fit_func(time[:truning_point], *p[0]))
    plt.plot(time[:truning_point],1e9*force[:truning_point])
    print(p[0])

    #%%
    tp_idx = np.where(indentation_dev_func(time)>0)[0][-1]
    print(time[tp_idx])
    tm = brenth(indentation_dev_func,0,time[-1],xtol=2.2e-22, rtol=8.881784197001252e-13)
    print(tm)
    plt.scatter(time, indentation_dev_func(time))
    #0.002528605318272992
    #%%
    plt.plot(indentation)
    plt.plot(indentation_func(time))
    plt.vlines(tp_idx, ymin = 0,ymax = 1e-6)
    #%%
    f_ting1 = ting(time, e0=100000,einf=10, alpha=0.5, 
            t_turning_point=tm,
            indentation_23_dev_func =indentation_23_dev_func,
            indentation_dev_func = indentation_dev_func)
    plt.plot(indentation[:tp_idx],f_ting1[:tp_idx])
    plt.plot(indentation[tp_idx:],f_ting1[tp_idx:])
    # f_ting2, t1 = ting(time, e0=100,einf=10, alpha=0.1, 
    #         t_turning_point=tm,
    #         indentation_23_dev_func =indentation_23_dev_func,
    #         indentation_dev_func = indentation_dev_func)
    # plt.plot(f_ting2)
    #%%
    #%%
    p_ting = curve_fit(lambda t, e0,einf,alpha:1e9*ting(t, e0=e0,einf=einf, alpha=alpha, t_turning_point=tm,
        indentation_23_dev_func =indentation_23_dev_func,
        indentation_dev_func = indentation_dev_func),
        time[:1200], 1e9*force[:1200], 
        method="trf", p0=(100000,0,0.5) ,
        bounds=((0,0,0),(np.inf,np.inf,1)))[0]
    p_ting=p_ting
    #%%

    f_fit_ting = ting(time, e0=p_ting[0],einf=p_ting[1], alpha=p_ting[2], t_turning_point=tm,
        indentation_23_dev_func =indentation_23_dev_func,
        indentation_dev_func = indentation_dev_func)

    plt.plot(f_fit_ting[:1200])
    # plt.plot(indentation*10**(-2.5))
    # plt.plot(force[:1200])
    # %%
    tp_idx
    plt.plot(indentation, f_fit_ting)