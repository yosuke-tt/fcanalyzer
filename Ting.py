import numpy as np



#%%
if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt
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
    # plt.plot(deflection)
    # plt.plot(deflection_func(np.arange(len(deflection))))

    # # plt.plot(2*app_point_sp - len(deflection_base)+np.arange(len(deflection_base[app_point_sp:])),
    # #                 deflection_base[app_point_sp:][::-1])
    # print(np.argmax(deflection))
    # print(np.argmax(zsensor[diff_app_point:]))

    # plt.plot(deflection*4)
    # plt.plot(zsensor[diff_app_point:]-6)
    # plt.plot(zsensor[diff_app_point+5000:-3000], deflection_base[5000:-3000-diff_app_point])
    # plt.scatter(zsensor[diff_app_point+5700], deflection_base[5700],c="red")
    ret_cp = np.where(zsensor[app_point_sp_z:]<zsensor[diff_app_point+5700])[0][0]
    # plt.scatter(zsensor[app_point_sp_z+ret_cp], deflection_base[app_point_sp_def+ret_cp],c="red")
    # plt.plot(zsensor[diff_app_point+5700:-3000], deflection_base[5700:-3000-diff_app_point])
    # plt.scatter(zsensor[diff_app_point+5700], deflection_base[5700],c="red")
    # plt.plot(zsensor[diff_app_point+5700:app_point_sp_z+ret_cp], deflection_base[5700:app_point_sp_def+ret_cp],c="red")


    force_base = deflection_base[5700:app_point_sp_def+ret_cp]-deflection_base[5700]
    force      = force_base*300*0.1*1e-9
    # plt.plot(force)

    indentation = (zsensor[diff_app_point+5700:app_point_sp_z+ret_cp]-zsensor[diff_app_point+5700])*25e-6-force_base*300*1e-9
    ind_positive = indentation>0
    indentation = indentation[ind_positive]
    force = force[ind_positive]

    time = np.arange(0,len(indentation))*3e-6
    time_app = np.arange(0,len(indentation))*3e-6
    indentation_fit = np.polyfit(time, indentation**(3/2), deg=10)
    indentation_func = np.poly1d(indentation_fit)
    indentation_dev_func = indentation_func.deriv()
    plt.plot(indentation**(3/2))
    plt.plot(indentation_func(time),c="red")
    print(np.argmax(indentation))
    from properties import hertz_constant
    from afm import AFM 
    from properties import et_e0
    from boltzmanns_superposition_principle import boltz_mann_superposition_principle
    afm = AFM(radius=5e-6,invols=300, k=0.1,no_afm=3)
    hc = hertz_constant(afm)
    
    def fit_func(time, e0, einf, alpha):
        efunc = lambda t:hc*et_e0(t,e0=e0, einf=einf, alpha=alpha)
        f = [boltz_mann_superposition_principle(t, indentation_dev_func,efunc) for t in time]
        return f
    from scipy.optimize import curve_fit
    

    p = curve_fit(fit_func, time[:np.argmax(indentation)], force[:np.argmax(indentation)] )[0]

#%%
efunc = lambda t:hc*et_e0(t,e0=p[0], einf=p[1], alpha=p[2])
f = [boltz_mann_superposition_principle(t, indentation_dev_func,efunc) for t in time[:np.argmax(indentation)]]
plt.plot(time[:np.argmax(indentation)], force[:np.argmax(indentation)])
plt.plot(time[:np.argmax(indentation)], force[:np.argmax(indentation)])