
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


def get_cp(deflection):
    return np.where(deflection[:len(deflection)//2]<0.02)[0][-1]

def preprocessing(deflection, 
                  zsensor, 
                  get_cp,
                  invols=30, 
                  um_per_v=25e-6, 
                  sampling_t=3e-6):
    
    cp = get_cp(deflection)
    deflection-=deflection[cp]
    diff_top_idx = np.argmax(deflection) - np.argmax(zsensor)

    deflection = deflection[diff_top_idx:]
    zsensor = zsensor[:-diff_top_idx]

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
    indentation_func, indentation_dev_func, indentation_23_dev_func = gen_indentaion_func(time, indentation)


    return force,indentation_func, indentation_dev_func, indentation_23_dev_func

def gen_indentaion_func(time, indentaion, output_coeff = False):
    
    indentation_fit = np.polyfit(time, indentaion, deg=30)
    indentation_func = np.poly1d(indentation_fit)
    indentation_dev_func = indentation_func.deriv()
    indentation_23_dev_func = lambda t:3*indentation_dev_func(t)*indentation_func(t)**(1/2)/2    
    if not output_coeff:
        return indentation_func, indentation_dev_func, indentation_23_dev_func
    else:
        return indentation_func, indentation_dev_func, indentation_23_dev_func
    


#%%

def boltz_mann_superposition_principle(start_t, end_t, t_now,
                                    delta_dev:callable,
                                    property_func:callable,
                                    ):

    def _integral_inner(g):
        return delta_dev(g)*property_func(t_now-g)
    return integrate.quad(_integral_inner, start_t, end_t)[0]

def searching_t1(time_ret, top_t,
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
            t1 = brenth(lambda t1:ret_integrate+boltz_mann_superposition_principle(t1,top_t,t,
                                                                                   indentation_dev_func,
                                                                                   efunc),
                0,t1_pre, xtol=2.2e-23, rtol=8.881784197001252e-16)

        except Exception as e:
            break
            
        t1s[i]+=t1
        t1_pre=t1
    return t1s
T_DASH=1/50000
def ting(time, properties_params,
         tp_idx, top_t,
         indentation_dev_func,
         indentation_23_dev_func,
         efunc_p):

    efunc = lambda time:efunc_p(time,*properties_params)
    t1s = searching_t1(time[tp_idx:],
                       top_t,
                        indentation_dev_func,
                        efunc)
    app_int = [boltz_mann_superposition_principle(0, t, t,
                                                indentation_23_dev_func,
                                                efunc) 
                for t in time[:tp_idx] ]

    ret_int = np.array([boltz_mann_superposition_principle(0, t, t,
                                                           indentation_23_dev_func,
                                                           efunc) 
            for t in t1s ])

    ting_curve=np.hstack([app_int,ret_int])
    return ting_curve


#%%

if __name__ =="__main__":
    zsensor = np.load("./zsensor.npy")
    deflection = np.load("./deflection.npy")

    force, indentation_func, indentation_dev_func, indentation_23_dev_func = preprocessing(deflection, 
                  zsensor, 
                  get_cp)



    time = np.arange(0,len(force))*3e-6


    tp_idx = np.where(indentation_dev_func(time)>0)[0][-1]

    roots = indentation_dev_func.roots
    tm = roots[np.isreal(roots)][0]

    afm = AFM(radius=5e-6,invols=30, k=0.1,no_afm=3)
    hc = hertz_constant(afm)

    top_t = brenth(indentation_dev_func,
                    0,
                    time[-1],
                    xtol=2.2e-23,
                    rtol=8.881784197001252e-16)


    efunc = lambda t, e1, einf, alpha :hc*et_e1(t,e1=e1, einf=einf, alpha=alpha)

    ting_fit_func = lambda time, e1 ,einf, alpha:1e9*ting(time, (e1, einf, alpha),
                                                    tp_idx,top_t,
                                                    indentation_dev_func,
                                                    indentation_23_dev_func,
                                                    efunc)


    p=curve_fit(ting_fit_func,time[:1400], 1e9*force[:1400],
                p0=(420,10,0.1),bounds=((0,0,0),(10000,100,1)) )[0]
    ting_res = ting_fit_func(time,*p)
    
    plt.plot(force*1e9)
    plt.title(p)
    plt.plot(ting_res)
    print(p)
    p=curve_fit(ting_fit_func,time[:1400], 1e9*force[:1400],
                p0=(100,0,0.9),bounds=((0,0,0),(1000,1,1)) )[0]
    print(p)
    ting_res = ting_fit_func(time,*p)
    
    plt.plot(force*1e9)
    plt.title(p)
    plt.plot(ting_res)


    efunc = lambda t, e1, alpha :hc*et_e1(t,e1=e1, einf=0, alpha=alpha)

    ting_fit_func = lambda time, e1 ,alpha:1e9*ting(time, (e1,  alpha),
                                                    tp_idx,top_t,
                                                    indentation_dev_func,
                                                    indentation_23_dev_func,
                                                    efunc)


    p=curve_fit(ting_fit_func,time[:1400], 1e9*force[:1400],
                p0=(420,0.1),bounds=((0,0),(10000,1)) )[0]
    ting_res = ting_fit_func(time,*p)
    print(p)
    plt.plot(force*1e9)
    plt.title(p)
    plt.plot(ting_res)


    efunc = lambda t, e0, einf, alpha :hc*et_e0(t,e0=e0, einf=einf, alpha=alpha)

    ting_fit_func = lambda time, e0 ,einf, alpha:1e9*ting(time, (e0, einf, alpha),
                                                    tp_idx,top_t,
                                                    indentation_dev_func,
                                                    indentation_23_dev_func,
                                                    efunc)


    p=curve_fit(ting_fit_func,time[:1400], 1e9*force[:1400],
                p0=(420,10,0.1),bounds=((0,0,0),(10000,100,1)) )[0]
    ting_res = ting_fit_func(time,*p)
    print(p)
    plt.plot(force*1e9)
    plt.title(p)
    plt.plot(ting_res)

    p=curve_fit(ting_fit_func,time[:1400], 1e9*force[:1400],
                p0=(100,0,0.9),bounds=((0,0,0),(1000,1,1)) )[0]
    ting_res = ting_fit_func(time,*p)
    print(p)
    plt.plot(force*1e9)
    plt.title(p)
    plt.plot(ting_res)
    
    efunc = lambda t, e0, alpha :hc*et_e0(t,e0=e0, einf=0, alpha=alpha)

    ting_fit_func = lambda time, e0 , alpha:1e9*ting(time, (e0, alpha),
                                                    tp_idx,top_t,
                                                    indentation_dev_func,
                                                    indentation_23_dev_func,
                                                    efunc)


    p=curve_fit(ting_fit_func,time[:1400], 1e9*force[:1400],
                p0=(420,0.1),bounds=((0,0),(10000,1)) )[0]
    print(p)
    ting_res = ting_fit_func(time,*p)
    plt.plot(force*1e9)
    plt.title(p)
    plt.plot(ting_res)

# %%
