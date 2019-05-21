import numpy as np
import matplotlib.pyplot as plt
import scal_power_spectra
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d

l_vec,cl_tt, cl_ee, cl_te, cl_tt_lens, cl_ee_lens, cl_bb_lens, cl_te_lens, cl_kk=scal_power_spectra.spectra()
clspl=spline1d(l_vec, cl_tt)
clspl_lens=spline1d(l_vec, cl_tt_lens)
ldcdl_spl=spline1d(l_vec, l_vec*np.gradient(cl_tt))

lmax=8192
delta_l=2
n=lmax/delta_l*2

lx=np.arange(-lmax,lmax, delta_l)
ly=np.arange(-lmax,lmax, delta_l)

lxs, lys=np.meshgrid(lx, ly)

l=np.sqrt(lxs**2+lys**2)



cos_2theta_grid=(lxs**2-lys**2)/(lxs.astype(float)**2+lys**2)
sin_2theta_grid=(2*lxs*lys)/(lxs.astype(float)**2+lys**2)
cos_2theta_grid[n/2,n/2]=1
sin_2theta_grid[n/2,n/2]=1

clgrid=np.zeros(l.shape)
clgridlens=np.zeros(l.shape)
ldcdl_grid=np.zeros(l.shape)

for i,l1 in enumerate(l[0,:]):
    clgrid[:,i]=clspl(l[:,i])
    clgridlens[:,i]=clspl_lens(l[:,i])
    ldcdl_grid[:,i]=ldcdl_spl(l[:,i])
    
plt.imshow(ldcdl_grid)
plt.colorbar()
plt.show()
    
ctheta=np.fft.ifft2(clgrid)  
plt.imshow(np.fft.ifft2(clgrid))
    
    