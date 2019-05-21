#run this second to create the filters that are used to perform the reconstruction
#This code takes nside (number of pixels per side in map) and fov_deg (field of view in degrees) as command-line arguments
#NB! Change working_dir variable in this code and in scal_power_spectra.py
#Make sure your working directory contains new_scalCls.dat and new_lensedCls.dat (output files from CAMB with spectra), as well as a folder called Filters

# Pass arguments as follows import sys ; sys.argv=['make_real_space_filter.py','2048','20'] ; execfile("make_real_space_filter.py")
# For MATHEW MAPS: Pass arguments as follows import sys ; sys.argv=['make_real_space_filter.py','1200','20'] ; execfile("make_real_space_filter.py")
# Folders to Create in Working Dir for Full Code Run: Filters LensingField Maps NoisePlots Plots Reconstructions Spectra

import random
import numpy as np
from math import *
import pylab
#from IPython.Shell import IPShellEmbed
import spline
import smooth
import matplotlib.pyplot as plt
import map_making_functions
import sys
import scal_power_spectra
import os
import time

from scipy.interpolate import InterpolatedUnivariateSpline as spline1d

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import inputs

# USING nside = 2048 and 20 deg

args=['make_real_space_filter.py','1024','20']
nside=int(args[1])      #Jet had 512. Heather changed to make ACT-like 2048 nside and 20 fov_deg
fov_deg=float(args[2])

print "inputs", nside, fov_deg

# STANDARDISE TO SET THESE CENTRALLY - KAVI
# NOTE THAT SIGMA_AM is BEAM FWHM - KAVI

#set these:
spec='tt' # ee te eb tb
exp='advact' # 'advact'   #'planck' or 'ref  # Need for naming the filter at the end. This is the experiment for which the later parameters are relevent and need to be adjusted
add_foregrounds=False


if (exp[0:3]=='ref'):
    working_dir=inputs.working_dir_base + 'ref' + '/fov'+str(int(fov_deg))
    if not os.path.exists(working_dir):
		os.makedirs(working_dir)
else:
    working_dir=inputs.working_dir_base+exp+'/fov'+str(int(fov_deg))
    if not os.path.exists(working_dir):
		os.makedirs(working_dir)

working_dir_Filter=inputs.working_dir_base+exp+'/fov'+str(int(fov_deg))+'/Filters'
if not os.path.exists(working_dir_Filter):
	os.makedirs(working_dir_Filter)
	
lmax=9999       #what CAMB stuff goes up to (2 to 10000 ignoring 10100)
lmax2=40000     #a bit over what we need for ACT resolution (33000ish)

l_vec,cl_tt, cl_ee, cl_te, cl_tt_lens, cl_ee_lens, cl_bb_lens, cl_te_lens, cl_kk=scal_power_spectra.spectra()


#returns d(cs)/d(ls) using 2-sided finite difference
#note:last 400 values will be dodgy coz trying to read past end of array but we only need up to 33 000 not 40 000 so I'm not going to worry to fix that
def diffSpectraManual(ls, cs):
    cspl=spline1d(ls, cs)
    dcdl=np.zeros(ls.shape[0])
    for i in range (ls.shape[0]):
        dcdl[i]=(cspl(ls[i]*1.01)-cspl(ls[i]*0.99))/(ls[i]*0.02)
    return dcdl



#main

print spec
print 'fov_deg= '+str(fov_deg)

#Noise stuff should match make_maps_with_noise
t_cmb=2.728
if exp=='planck':
    sigma_am=7.
    delta_t_am=27.
    delta_p_am=40*sqrt(2)

if exp=='advact':   #150 GHz channel, from Henderson et al 2015
    sigma_am=1.4
    delta_t_am=7.   #microkelvin-arcmin
    delta_p_am=delta_t_am*sqrt(2)


if exp=='act':
    delta_t_am=15
    delta_p_am=delta_t_am*sqrt(2)
    sigma_am=1.5

if exp=='ref':  # NOTE: THIS WAS UPDATED FOR MATHEW MAPS TO SPECS PROVIDED - delta_am is in uKarcmin
    sigma_am=1.5 # 1.4  #1.#60*fov_deg/nside #1. #Should be 7 for Planck, ~1 for ACT. But in reality, this should be governed by the pixel scale of your map. This may need some thinking about/adjustment of map sizes/pixel scales to match the experiment
    delta_t_am= 1.5 # 2.      #1.     #sigma_am/(60*15./512.) #(60*15./512.)/sigma_am#*0.5 #0.01  #Heather changed # Should be 27 for Planck, ~1 for ACT I think (check this)
    delta_p_am=delta_t_am*sqrt(2)

# KM UPDATED REF TO RUN ON SIMULATED ACTPOL MAPS
 
elif (exp[0:3]=='ref'):
    delta_t_am=float(exp[3:])
    delta_p_am=delta_t_am*sqrt(2)
    sigma_am=1.

exp_print = exp + '_noise_'+str(round(delta_t_am,2))+'_'+'beam_'+str(round(sigma_am,2))
spec_print = spec + '_'+str(nside)+'_'+str(int(fov_deg))

sigma_rad=(sigma_am)*(pi/(60*180))
delta_t=delta_t_am*(pi/(60*180))
delta_p=delta_p_am*(pi/(60*180))   # Should be 40*sqrt(2) for Planck, for ACTPol I don't know

n_l_tt=(delta_t/t_cmb)**2*np.exp(l_vec*(l_vec+1)*sigma_rad**2/(8*log(2)))   #(delta_t/t_cmb)
n_l_ee=(delta_p/t_cmb)**2*np.exp(l_vec*(l_vec+1)*sigma_rad**2/(8*log(2)))   #(delta_p/t_cmb)
n_l_bb=n_l_ee

plt.loglog(l_vec, l_vec**2*cl_tt)
plt.loglog(l_vec, l_vec**2*n_l_tt)
plt.show()

plt.figure()
n_l_tt=scal_power_spectra.getNoiseTT(l_vec, 'advact')
plt.loglog(l_vec, l_vec**2*cl_tt)
plt.loglog(l_vec, l_vec**2*n_l_tt)
plt.show()

# exit()


"""
datafile=np.zeros((3999, 4))
datafile[:,0]=l_vec[0:3999]
datafile[:,1]=cl_tt_lens[0:3999]
datafile[:,2]=cl_tt[0:3999]
datafile[:,3]=n_l_tt[0:3999]
np.savetxt(working_dir+'/tt_datafile', datafile)

datafile[:,3]=scal_power_spectra.getNoiseTT(l_vec[0:3999],'planck')
np.savetxt(working_dir+'/tt_datafile2', datafile)
"""
#trying to account for foregrounds
if exp=='ref':
    if add_foregrounds:
        cib_poisson=l_vec**2*1e-6#900000
        plt.plot(l_vec, l_vec*(l_vec+1)/(2*np.pi)*cl_tt, label='l(l+1)c/2pi')
        #plt.plot(l_vec, l_vec*(l_vec+1)/(2*np.pi)*n_l_tt, label='l(l+1)n/2pi')
        plt.plot(l_vec, l_vec*(l_vec+1)/(2*np.pi)*n_l_tt2, label='l(l+1)n/2pi')
        plt.plot(l_vec, cib_poisson, label='CIB poisson')
        plt.plot(l_vec, cib_poisson+l_vec*(l_vec+1)/(2*np.pi)*n_l_tt2, label='CIB poisson+noise')
        #plt.xscale('log')
        plt.xlim(0,10000)
        plt.ylim(1e-2,1e4)
        plt.yscale('log')
        plt.legend()
        plt.show()
    
        n_l_tt=cib_poisson/(l_vec*(l_vec+1)/(2*np.pi))+n_l_tt2
    else:
        n_l_tt=scal_power_spectra.getNoiseTT(l_vec, 'ref')



#the following functions make the filters in harmonic space. Currently, the convolution in Bucher et al is applied as a product in Fourier space instead of a convolution in real space, because our simulated maps are nice and periodic so we can take the Fourier transform easily

# For consistency in the code we have a null filter for the convergence in the TB and EB case
def make_null_filt(l_vec):
  zero=np.zeros(shape=(l_vec.size))
  filt_vec=zero
  filt_fun=spline.Spline(l_vec,filt_vec)
  return (filt_vec,filt_fun)


def make_tt_conv_filt(l_vec,cl_tt,cl_tt_lens,n_l_tt):
  a=(cl_tt/(cl_tt_lens+n_l_tt)**2)
  print 'type of thing giving overflow', type(a[0])
  lnCl=np.log(cl_tt)
  lnl=np.log(l_vec)
  d=np.zeros(shape=(l_vec.size))
  dx=np.zeros(shape=(l_vec.size))
  dy=np.zeros(shape=(l_vec.size))
  dcdl=diffSpectraManual(l_vec, cl_tt) #should it be cl_tt_lens??
  dlncdlnl=dcdl*l_vec/cl_tt             #cl_tt_lens??
  d=dlncdlnl+2.
  g=a*d
  
  #for normalisation: 
  #are we assuming radial symmetry (DALIAN)
  N_integrand=g*cl_tt*d*l_vec/(2*pi)
  N=np.trapz(N_integrand)
  print N

  filt_vec=g/N

  filt_vec[l_vec.size-1]=0.
  filt_fun=spline.Spline(l_vec,filt_vec)
  plt.figure()
  plt.plot(l_vec[0:lmax], filt_vec[0:lmax])
  plt.title('Convergence Filter in Harmonic Space')
  plt.xlabel('l')
  plt.ylabel('$K^{TT}_{\kappa_0}(l)$')
  plt.savefig(working_dir+'/Filters/HS_filter_1d'+'_'+spec_print+'_conv_'+exp_print+'.png')
  plt.show()
  
  return (filt_vec,filt_fun)


def make_tt_shear_filt(l_vec,cl_tt,cl_tt_lens,n_l_tt):
  
  a=(cl_tt/(cl_tt_lens+n_l_tt)**2)
  lnCl=np.log(cl_tt)
  lnl=np.log(l_vec)
  d=np.zeros(shape=(l_vec.size))
  dx=np.zeros(shape=(l_vec.size))
  dy=np.zeros(shape=(l_vec.size))

  dcdl=diffSpectraManual(l_vec, cl_tt) #should it be cl_ee_lens??
  d=dcdl*l_vec/cl_tt             #gives dlncdlnl # use cl_ee_lens??
  g=a*d
  
  N_integrand=0.5*g*cl_tt*d*l_vec/(2*pi)
  N=np.trapz(N_integrand)
  
  filt_vec=g/N

  filt_vec[l_vec.size-1]=0.
  filt_fun=spline.Spline(l_vec,filt_vec)
  plt.figure()
  plt.plot(l_vec[0:lmax], filt_vec[0:lmax])
  plt.title('Shear Filter in Harmonic Space')
  plt.xlabel('l')
  plt.ylabel('$K^{TT}_{\gamma}(l)$')
  plt.savefig(working_dir+'/Filters/HS_filter_1d'+'_'+spec_print+'_shear_'+exp_print+'.png')
  plt.show()

  return (filt_vec,filt_fun)


def make_ee_conv_filt(l_vec,cl_ee,cl_ee_lens,n_l_ee):
  
  a=(cl_ee/(cl_ee_lens+n_l_ee)**2)
  lnCl=np.log(cl_ee)
  lnl=np.log(l_vec)
  d=np.zeros(shape=(l_vec.size))
  dx=np.zeros(shape=(l_vec.size))
  dy=np.zeros(shape=(l_vec.size))
  for k in range(l_vec.size-1):
    if cl_ee[k]<0:
      lnCl[k]=0.

  dcdl=diffSpectraManual(l_vec, cl_ee) #should it be cl_ee_lens??
  dlncdlnl=dcdl*l_vec/cl_ee             #cl_ee_lens??
  d=dlncdlnl+2.
  g=a*d

  N_integrand=g*cl_ee*d*l_vec/(2*pi)
  N=np.trapz(N_integrand)

  filt_vec=g/N

  filt_vec[l_vec.size-1]=0.
  filt_vec[l_vec.size-2]=0.
  filt_fun=spline.Spline(l_vec,filt_vec)
  
  plt.plot(l_vec[0:lmax], filt_vec[0:lmax])
  plt.title('Convergence Filter in Harmonic Space')
  plt.xlabel('l')
  plt.ylabel('$K^{EE}_{\kappa_0}(l)$')
  plt.savefig(working_dir+'/Filters/HS_filter_1d'+'_'+spec_print+'_conv_'+exp_print+'.png')
  plt.show()

  return (filt_vec,filt_fun)


def make_ee_shear_filt(l_vec,cl_ee,cl_ee_lens,n_l_ee):
  
  a=(cl_ee/(cl_ee_lens+n_l_ee)**2)
  lnCl=np.log(cl_ee)
  lnl=np.log(l_vec)
  d=np.zeros(shape=(l_vec.size))
  dx=np.zeros(shape=(l_vec.size))
  dy=np.zeros(shape=(l_vec.size))

  
  dcdl=diffSpectraManual(l_vec, cl_ee) #should it be cl_ee_lens??
  d=dcdl*l_vec/cl_ee             #gives dlncdlnl # use cl_ee_lens??
  #filt_vec=a*d
  g=a*d
  N_integrand=0.5*g*cl_ee*d*l_vec/(2*pi)
  N=np.trapz(N_integrand)

  filt_vec=g/N

  filt_vec[l_vec.size-1]=0.
  filt_vec[l_vec.size-2]=0.
  filt_fun=spline.Spline(l_vec,filt_vec)

  plt.plot(l_vec[0:lmax], filt_vec[0:lmax])
  plt.title('Shear Filter in Harmonic Space')
  plt.xlabel('l')
  plt.ylabel('$K^{EE}_{\gamma}(l)$')
  plt.savefig(working_dir+'/Filters/HS_filter_1d'+'_'+spec_print+'_shear_'+exp_print+'.png')
  plt.show()

  return (filt_vec,filt_fun)



def make_te_conv_filt(l_vec,cl_te, cl_te_lens, cl_ee_lens,n_l_ee, cl_tt_lens, n_l_tt):
    
    a=1/((cl_te_lens)**2+(cl_ee_lens+n_l_ee)*(cl_tt_lens+n_l_tt))
    lnl=np.log(l_vec)

    dcdl=diffSpectraManual(l_vec, cl_te) #should it be cl_ee_lens??
    dcdlnl=dcdl*l_vec             #cl_ee_lens??
    d=dcdlnl+2.*cl_te
    g=a*d
    
    N_integrand=g*d*l_vec/(2*pi)
    N=np.trapz(N_integrand)
            
    filt_vec=g/N

    filt_vec[l_vec.size-1]=0.
    filt_vec[l_vec.size-2]=0.
    filt_fun=spline.Spline(l_vec,filt_vec)
    
    plt.plot(l_vec[0:lmax], filt_vec[0:lmax])
    plt.title('Convergence Filter in Harmonic Space')
    plt.xlabel('l')
    plt.ylabel('$K^{TE}_{\kappa_0}(l)$')
    plt.savefig(working_dir+'/Filters/HS_filter_1d'+'_'+spec_print+'_conv_'+exp_print+'.png')
    plt.show()
    return (filt_vec,filt_fun)

def make_te_shear_filt(l_vec,cl_te, cl_te_lens, cl_ee_lens,n_l_ee, cl_tt_lens, n_l_tt):
    
    a=1/((cl_te_lens)**2+(cl_ee_lens+n_l_ee)*(cl_tt_lens+n_l_tt))
    lnl=np.log(l_vec)

    dcdl=diffSpectraManual(l_vec, cl_te) #should it be cl_ee_lens??
    dcdlnl=dcdl*l_vec             #cl_ee_lens??
    d=dcdlnl
    g=a*d
    
    N_integrand=0.5*g*d*l_vec/(2*pi)
    N=np.trapz(N_integrand)
    
    filt_vec=g/N
    

    filt_vec[l_vec.size-1]=0.
    filt_vec[l_vec.size-2]=0.
    filt_fun=spline.Spline(l_vec,filt_vec)
    
    plt.plot(l_vec[0:lmax], filt_vec[0:lmax])
    plt.title('Shear Filter in Harmonic Space')
    plt.xlabel('l')
    plt.ylabel('$K^{TE}_{\gamma}(l)$')
    plt.savefig(working_dir+'/Filters/HS_filter_1d'+'_'+spec_print+'_shear_'+exp_print+'.png')
    plt.show()
    return (filt_vec,filt_fun)


def make_tb_shear_filt(l_vec, cl_te,cl_tt_lens,cl_bb_lens,n_l_tt,n_l_bb):
    filt_vec=cl_te/((cl_tt_lens+n_l_tt)*(cl_bb_lens+n_l_bb))
    N_integrand=filt_vec*cl_te*l_vec/(2*pi)
    N=np.trapz(N_integrand)
    filt_vec/=N
  
    filt_fun=spline.Spline(l_vec,filt_vec)
  
    plt.plot(l_vec[0:lmax], filt_vec[0:lmax])
    plt.title('Shear Filter in Harmonic Space')
    plt.xlabel('l')
    plt.ylabel('$K^{TB}_{\gamma}(l)$')
    plt.savefig(working_dir+'/Filters/HS_filter_1d'+'_'+spec_print+'_shear_'+exp_print+'.png')
    plt.show()
    return(filt_vec, filt_fun)


def make_eb_shear_filt(l_vec, cl_ee,cl_ee_lens,cl_bb_lens,n_l_ee,n_l_bb):
    filt_vec=cl_ee/((cl_ee_lens+n_l_ee)*(cl_bb_lens+n_l_bb))
    N_integrand=filt_vec*cl_ee*l_vec/(2*pi)
    N=np.trapz(N_integrand)
    filt_vec/=N

    plt.plot(l_vec[0:lmax], filt_vec[0:lmax])
    plt.title('Shear Filter in Harmonic Space')
    plt.xlabel('l')
    plt.ylabel('$K^{EB}_{\gamma}(l)$')
    plt.savefig(working_dir+'/Filters/HS_filter_1d'+'_'+spec_print+'_shear_'+exp_print+'.png')
    plt.show()

    filt_fun=spline.Spline(l_vec,filt_vec)

    return(filt_vec, filt_fun)


##############################
### Make the filters #########
##############################

if spec=='tt':
  filt_vec_conv,filt_fun_conv=make_tt_conv_filt(l_vec,cl_tt,cl_tt_lens,n_l_tt)
  filt_vec_shear,filt_fun_shear=make_tt_shear_filt(l_vec,cl_tt,cl_tt_lens,n_l_tt)

if spec=='ee':
  filt_vec_conv,filt_fun_conv=make_ee_conv_filt(l_vec,cl_ee,cl_ee_lens,n_l_ee)
  filt_vec_shear,filt_fun_shear=make_ee_shear_filt(l_vec,cl_ee,cl_ee_lens,n_l_ee)


if spec=='te':
    filt_vec_conv,filt_fun_conv=make_te_conv_filt(l_vec,cl_te, cl_te_lens, cl_ee_lens,n_l_ee, cl_tt_lens, n_l_tt)
    filt_vec_shear,filt_fun_shear=make_te_shear_filt(l_vec,cl_te, cl_te_lens, cl_ee_lens,n_l_ee, cl_tt_lens, n_l_tt)

# For EB and TB, there is no convergence estimator
if spec=='tb':
  filt_vec_conv, filt_fun_conv=make_null_filt(l_vec)
  filt_vec_shear, filt_fun_shear=make_tb_shear_filt(l_vec, cl_te, cl_tt_lens, cl_bb_lens,n_l_tt, n_l_bb)

if spec=='eb':
  filt_vec_conv, filt_fun_conv=make_null_filt(l_vec)
  print "filt_vec_conv size", filt_vec_conv.size
  filt_vec_shear, filt_fun_shear=make_eb_shear_filt(l_vec, cl_ee, cl_ee_lens, cl_bb_lens, n_l_ee, n_l_bb)

l_array=map_making_functions.make_l_array(nside, fov_deg)
lx_array, ly_array=map_making_functions.make_lxy_arrays(nside,fov_deg)
cos_2theta_array=(lx_array**2-ly_array**2)/(lx_array**2+ly_array**2)
sin_2theta_array=(2*lx_array*ly_array)/(lx_array**2+ly_array**2)

def make_filt(filt_fun,nside, l_array):
  filt=np.zeros(shape=(nside,nside),dtype=float)
  for i in range(nside):
    for j in range(nside):
      if l_array[i,j]>0:
        filt[i,j]=filt_fun(l_array[i,j])
  return(filt)


filt_conv=make_filt(filt_fun_conv,nside, l_array)
filt_shear=make_filt(filt_fun_shear,nside, l_array)
if spec=='tt' or spec=='ee' or spec=='te':
  filter_plus=cos_2theta_array*filt_shear
  filter_cross=sin_2theta_array*filt_shear
if spec=='tb' or spec=='eb':
  filter_plus=sin_2theta_array*filt_shear
  filter_cross=-cos_2theta_array*filt_shear

filter_plus[0,0]=0.
filter_cross[0,0]=0.

plt.figure()
#3d plot - remove
plt.imshow(np.fft.fftshift(filt_conv))
plt.title('Convergence Filter in Harmonic Space')
plt.savefig(working_dir+'/Filters/HS_filter_2d'+'_'+spec_print+'_conv_'+exp_print+'.png')
plt.show()

plt.figure()
plt.imshow(np.fft.fftshift(filter_plus))
plt.title('Shear Plus Filter in Harmonic Space')
plt.savefig(working_dir+'/Filters/HS_filter_2d'+'_'+spec_print+'_shear_'+exp_print+'.png')
plt.show()

filter_conv_RS=np.fft.fftshift(np.real(np.fft.ifft2(filt_conv))) #real space filter
filter_plus_RS=np.fft.fftshift(np.real(np.fft.ifft2(filter_plus)))
filter_cross_RS=np.fft.fftshift(np.real(np.fft.ifft2(filter_cross)))

print filter_conv_RS
print 'RS shape', filter_conv_RS.shape, filter_plus_RS.shape, filter_cross_RS.shape

plt.figure()
max=0.000005
plt.imshow((filter_conv_RS)[nside/2-nside/8:nside/2+nside/8, nside/2-nside/8:nside/2+nside/8], origin='lower', extent=[0,fov_deg/4.,0,fov_deg/4.], vmin=-max, vmax=max, cmap='RdBu')
plt.colorbar()
plt.title('Convergence Filter in Real Space')
plt.savefig(working_dir+'/Filters/RS_filter_2d'+'_'+spec_print+'_conv_'+exp_print+'.png')
plt.show()


plt.figure()
plt.imshow((filter_plus_RS)[nside/2-nside/8:nside/2+nside/8, nside/2-nside/8:nside/2+nside/8], extent=(-1*fov_deg/8, fov_deg/8, -1*fov_deg/8, fov_deg/8), cmap='RdBu')
plt.colorbar()
plt.title('Shear Plus Filter in Real Space')
plt.savefig(working_dir+'/Filters/RS_filter_2d'+'_'+spec_print+'_shear_plus_'+exp_print+'.png')
plt.show()

plt.figure()
plt.imshow((filter_cross_RS)[nside/2-nside/8:nside/2+nside/8, nside/2-nside/8:nside/2+nside/8], extent=(-1*fov_deg/8, fov_deg/8, -1*fov_deg/8, fov_deg/8), cmap='RdBu')
plt.colorbar()
plt.title('Shear Cross Filter in Real Space')
plt.savefig(working_dir+'/Filters/RS_filter_2d'+'_'+spec_print+'_shear_plus_'+exp_print+'.png')
plt.show()

plt.figure()
theta=np.arange(-nside/2,nside/2, 1)/float(nside)*fov_deg
plt.plot(theta[nside/2:], np.abs((filter_conv_RS)[nside/2,nside/2:]))
plt.semilogy()
plt.title(spec.upper()+' Convergence Filter in Real Space')
plt.xlabel('x [$^\circ$]')
plt.ylabel('$K_{\kappa_0}$')
plt.savefig(working_dir+'/Filters/RS_filter_1d'+'_'+spec_print+'_conv_'+exp_print+'.png')
plt.show()

plt.figure()

filt_conv_to_plot=np.zeros((nside/2,2))
filt_conv_to_plot[:,0]=theta[nside/2:]
filt_conv_to_plot[:,1]=(filter_conv_RS)[nside/2,nside/2:]

filt_conv_to_plot_HS=np.zeros((nside,2))
filt_conv_to_plot_HS[:,0]=l_array[0,:]
filt_conv_to_plot_HS[:,1]=filt_conv[0,:]

filt_shear_to_plot=np.zeros((nside/2,2))
filt_shear_to_plot[:,0]=theta[nside/2:]

filt_shear_to_plot_HS=np.zeros((nside,2))
filt_shear_to_plot_HS[:,0]=l_array[0,:]

if spec=='tt' or spec=='ee' or spec=='te':
    #plt.plot(theta[nside/2:nside/2+nside/8], (filter_plus_RS)[nside/2, nside/2:nside/2+nside/8])
    plt.plot(theta[nside/2:], np.abs((filter_plus_RS)[nside/2, nside/2:]))
    filt_shear_to_plot[:,1]=(filter_plus_RS)[nside/2, nside/2:]
    filt_shear_to_plot_HS[:,1]=filter_plus[0,:]
else:
    #plt.plot(theta[nside/2:nside/2+nside/8], -1*(filter_cross_RS)[nside/2, nside/2:nside/2+nside/8])
    plt.plot(theta[nside/2:], np.abs(-1*(filter_cross_RS)[nside/2, nside/2:]))
    filt_shear_to_plot[:,1]=-1*(filter_cross_RS)[nside/2, nside/2:]
    filt_shear_to_plot_HS[:,1]=-1*filter_cross[0,:]
plt.title(spec.upper()+' Shear Filter in Real Space')
plt.semilogy()
plt.xlabel('x [$^\circ$]')
plt.ylabel('$K_{\gamma}$')
plt.savefig(working_dir+'/Filters/RS_filter_1d'+'_'+spec_print+'_shear_plus_'+exp_print+'.png')
plt.show()



if add_foregrounds:
    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_plus_filt_'+'_with_cib_poisson',filter_plus)
    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_cross_filt_'+'_with_cib_poisson',filter_cross)
    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_conv_filt_'+'_with_cib_poisson',filt_conv)

    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_plus_filt_RS_'+'_with_cib_poisson',filter_plus)
    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_cross_filt_RS_'+'_with_cib_poisson',filter_cross)
    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_conv_filt_RS_'+'_with_cib_poisson',filt_conv)
    
    #np.savetxt(working_dir+'/Filters/'+exp+'_'+spec+'_shear_plus_filt_RS_'+str(fov_deg)+'_'+str(nside)+'_with_cib_poisson',filter_plus_RS)
    #np.savetxt(working_dir+'/Filters/'+exp+'_'+spec+'_shear_cross_filt_RS_'+str(fov_deg)+'_'+str(nside)+'_with_cib_poisson',filter_cross_RS)
    #np.savetxt(working_dir+'/Filters/'+exp+'_'+spec+'_conv_filt_RS_'+str(fov_deg)+'_'+str(nside)+'_with_cib_poisson',filter_conv_RS)
else:
    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_plus_filt',filter_plus)
    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_cross_filt',filter_cross)
    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_conv_filt',filt_conv)

    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_plus_filt_RS',filter_plus_RS)
    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_cross_filt_RS',filter_cross_RS)
    np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_conv_filt_RS',filter_conv_RS)

    #np.savetxt(working_dir+'/Filters/'+exp+'_'+spec+'_shear_plus_filt_RS_'+str(fov_deg)+'_'+str(nside),filter_plus_RS)
    #np.savetxt(working_dir+'/Filters/'+exp+'_'+spec+'_shear_cross_filt_RS_'+str(fov_deg)+'_'+str(nside),filter_cross_RS)
    #np.savetxt(working_dir+'/Filters/'+exp+'_'+spec+'_conv_filt_RS_'+str(fov_deg)+'_'+str(nside),filter_conv_RS)

np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_filt_to_plot_RS',filt_shear_to_plot)
np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_conv_filt_to_plot_RS',filt_conv_to_plot)

np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_filt_to_plot_HS',filt_shear_to_plot_HS)
np.savetxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_conv_filt_to_plot_HS',filt_conv_to_plot_HS)



