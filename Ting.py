
#%%
import numpy as np
import matplotlib.pyplot as plt


from scipy.optimize import curve_fit, brenth
from scipy import integrate  
from scipy.optimize import newton



from properties import hertz_constant
from afm import AFM 
from properties import et_e0, et_e1
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

zsensor = np.load("./zsensor.npy")
deflection = np.load("./deflection.npy")
# plt.plot(deflection)
cp = np.where(deflection[:len(deflection)//2]<0.02)[0][-1]
# plt.scatter(cp, deflection[cp])
deflection-=deflection[cp]
# plt.plot()
diff_maxidx = np.argmax(zsensor)-np.argmax(deflection)

ret_cp = np.argmax(zsensor)+np.where(zsensor[np.argmax(zsensor):]<zsensor[cp])[0][0]
# plt.plot(zsensor[cp-diff_maxidx:ret_cp-diff_maxidx])
# plt.plot(deflection[cp:ret_cp])
zsensor = zsensor[cp+diff_maxidx:ret_cp+diff_maxidx]-zsensor[cp+diff_maxidx]
deflection = deflection[cp:ret_cp]-deflection[cp]

force      = deflection*30*0.1*1e-9
indentation = (zsensor-zsensor[0])*25e-6-deflection*300*1e-9
# indentation = np.hstack([np.arange(1000),np.arange(1000,0,-1)])*1e-9
ind_positive = indentation>0
indentation = indentation[ind_positive]
force =force[ind_positive]
# force = force[ind_positive]

time = np.arange(0,len(indentation))*3e-6

indentation_fit = np.polyfit(time, indentation, deg=30)
indentation_func = np.poly1d(indentation_fit)
indentation_dev_func = indentation_func.deriv()
indentation_23_dev_func = lambda t:3*indentation_dev_func(t)*indentation_func(t)**(1/2)/2    

tp_idx = np.where(indentation_dev_func(time)>0)[0][-1]
tm = brenth(indentation_dev_func,
            0,
            time[-1],xtol=2.2e-22, rtol=8.881784197001252e-13)

afm = AFM(radius=5e-6,invols=30, k=0.1,no_afm=3)

hc = hertz_constant(afm)
#%%
def boltz_mann_superposition_principle(t,
                                    delta_dev:callable,
                                    property_func:callable = et_e1,
                                    start_t=0
                                    ):
    if start_t==t:
        return 0
    def _integral_inner(g):
        return delta_dev(g)*property_func(t-g)
    # fig,ax = plt.subplots(2,1)
    # ax[0].set_title("ret "+ str(t)+ "  "+str(start_t))
    # ax[0].plot(np.linspace(start_t ,t, 100),[delta_dev(g) for  g in np.linspace(start_t ,t, 100)])
    # ax[1].plot(np.linspace(start_t ,t, 100),[property_func(t-g) for  g in np.linspace(start_t ,t,100)])
    # plt.show()
    return integrate.quad(_integral_inner, start_t, t)[0]


def boltz_mann_superposition_principle_t1(t,tm,
                                    delta_dev:callable,
                                    property_func:callable = et_e1,
                                    start_t=0
                                    ):
    if start_t==t:
        return 0
    def _integral_inner(g):
        # print("app")
        # print(t, g, t-g)
        # print(delta_dev(g), property_func(t-g))
        return delta_dev(g)*property_func(t-g)

    return integrate.quad(_integral_inner, start_t, tm)[0]

def _app_integrate_t1(t, t1, tm, efunc):
    return boltz_mann_superposition_principle_t1(t, tm,
                                              indentation_dev_func, 
                                              efunc, 
                                              start_t=t1)
T_DASH=1/50000
def ting(time, e1, einf, alpha,
         tp_idx,top_t,
         indentation_dev_func,
         indentation_23_dev_func
         ):
    tm = time[tp_idx]
    def et_e1(t, e1, einf, alpha):
        return einf+(e1-einf)*(t+T_DASH)**(-alpha)
    efunc = lambda t:hc*et_e1(t,e1=e1, einf=einf, alpha=alpha)
    t1s=[]
    t1_pre=top_t

    k = 0
    for t in time[tp_idx:]:
        ret_integrate = boltz_mann_superposition_principle(t, 
                                                            indentation_dev_func,
                                                            efunc,
                                                            start_t=top_t)


        try:
            t1 = brenth(lambda t1:ret_integrate+_app_integrate_t1(t,t1,top_t,efunc),
                0,t1_pre, xtol=2.2e-23, rtol=8.881784197001252e-16)

        except Exception as e:
            # print(ret_integrate)
            # print(ret_integrate+_app_integrate_t1(t,0,tm, efunc))
            # print(ret_integrate+_app_integrate_t1(t,t1_pre,tm, efunc))
            # print(e)
            t1=0
        k+=1

        t1_pre=t1
        t1s.append(t1)
    app_int = [boltz_mann_superposition_principle(t, 
                                                indentation_23_dev_func,
                                                efunc, 
                                                start_t=0) 
                for t in time[:tp_idx] ]

    ret_int = np.array([boltz_mann_superposition_principle(t, indentation_23_dev_func, efunc, start_t=0) 
            for t in t1s ])
    ting_curve=np.hstack([app_int,ret_int])
    print(e1,einf,alpha)
    return ting_curve

top_t = brenth(indentation_dev_func,
    0,time[-1], xtol=2.2e-23, rtol=8.881784197001252e-16)


ting_fit_func = lambda time, e1,einf,alpha:1e9*ting(time, e1,einf,alpha,
                                                tp_idx+1,top_t,
                                                indentation_dev_func,
                                                indentation_23_dev_func)


p=curve_fit(ting_fit_func,time[:1400], 1e9*force[:1400],
            p0=(100,0,0.1),bounds=((0,0,0),(1000,100,1)) )[0]
ting_res = ting_fit_func(time,*p)
#%%
plt.plot(1e9*force[:1400])
plt.plot(ting_res)
plt.plot(indentation_func(time)*1e6)
print(np.argmax(indentation_func(time)*1e6))
#%%
# ff, t1s = ting_fit_func(time,*[80,79,1])
plt.plot(indentation_func(time)[:tp_idx],ff[:tp_idx])
plt.plot(indentation_func(time)[tp_idx:],ff[tp_idx:])


#%%
def et_e1(t, e1, einf, alpha):
    return einf+(e1-einf)*(t+T_DASH)**(-alpha)
efunc = lambda t:hc*et_e1(t,e1=e1, einf=einf, alpha=alpha)

ret_integrate = [boltz_mann_superposition_principle(t, 
                                                    indentation_dev_func,
                                                    efunc,
                                                    start_t=tm) for t in time[tp_idx+1:]]
plt.plot(ret_integrate)
#%%


#%%
time_ = time.copy()
p=curve_fit(ting_fit_func,time[:1400], force[:1400],
            p0=(100,0,0.1),bounds=((0,0,0),(1000,100,1)) )[0]
ting_res = ting_fit_func(time,*p)
#%%
ff = ting_fit_func(time,*[80,79,0])
#%%
# plt.plot(time[:1400],ting_res[:1400])
# plt.plot(time,indentation_23_dev_func(time))
plt.plot(time,indentation_func(time)*1e-3)
plt.plot(time,ff)
plt.plot(time,force)
#%%
ting_res

#%%
print(len(time),np.argmax(force),np.argmax(indentation_func(time)))
#%%
# from boltzmanns_superposition_principle import bsp_stress_relaxation_for_e0
e1 = 10
einf = 1
alpha = 0.1

# bsp_beta = bsp_stress_relaxation_for_e0(time[:tp_idx], e0=100,einf=0,alpha=0.9,
#                              t_trig=tm,k_exp_app=1/2,k_coeff_app=3/2)
def et_e1(t, e1, einf, alpha):
    return einf+(e1-einf)*(t)**(-alpha)
efunc = lambda t:hc*et_e1(t,e1=e1, einf=einf, alpha=alpha)
app_int = [boltz_mann_superposition_principle(t, 
                                            indentation_23_dev_func,
                                                efunc, 
                                                start_t=0) 
                for t in time[:tp_idx] ]
plt.plot(app_int)