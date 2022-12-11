#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%
fc = np.loadtxt("../data_000228/ForceCurve/ForceCurve_005.lvm")
deflection = fc[:len(fc)//2]
zsensor = fc[len(fc)//2:]


app_point_sp_z = np.argmax(zsensor)
app_point_sp_def = np.argmax(deflection)
diff_app_point = app_point_sp_z-app_point_sp_def
deflection=deflection[:-diff_app_point]
zsensor=zsensor[diff_app_point:]

point_idx = np.arange(len(zsensor))

zsensor_func = np.poly1d(np.polyfit(point_idx,
                                    zsensor, deg=3))
zsensor_dev_func = zsensor_func.deriv()

app_point = 10000
ret_point = 8000

x_fit  = np.hstack([
            point_idx[:app_point],
            point_idx[-ret_point:]
        ])

# x_fit  = np.hstack([
#             zsensor_dev_func(point_idx[:app_point]),
#             zsensor_dev_func(point_idx[-ret_point:])
#         ])
y_fit  = np.hstack([
            deflection[:app_point],
            deflection[-ret_point:]
        ])
# x_fit = zsensor_dev_func(point_idx[:app_point])
# y_fit = deflection[:app_point]


fit_vartual_deflection = np.poly1d(np.polyfit(x_fit, y_fit, deg=2))
sigmoid = lambda x,a,b,c,d: c/(1+np.exp(a*(x-b)))+d
p = curve_fit(sigmoid, x_fit, y_fit, p0 = (0.025,app_point_sp_def,-1,0))
plt.plot(deflection)
plt.plot(sigmoid(point_idx,*p[0]))

deflection_base =deflection - sigmoid(point_idx,*p[0])
plt.plot(deflection_base)

np.save("deflection",deflection_base)
np.save("zsensor",zsensor)

#%%
plt.plot(sigmoid(np.arange(-10,10,0.01),1,1,1,1))
#%%
plt.plot(deflection_base[:np.argmax(deflection_base)])
plt.plot(deflection_base[np.argmax(deflection_base):][::-1])