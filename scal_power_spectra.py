import random
import scipy as sp
import numpy as np
from math import *
import pylab
#from IPython.Shell import IPShellEmbed #commented by Heather
import spline
import smooth
import matplotlib.pyplot as plt
import map_making_functions
from scipy.stats import itemfreq #for combining spectra with same l
import sys
sys.path.insert(0, '/home/zahra/python_scripts/kSZ_21cm_signal')
import lin_kSZ_power_spec as lpkSZ

import inputs

working_dir=inputs.working_dir_base # '/Users/Heather/Dropbox/PolRealSp/Codes/HeatherVersion'

lmax=9999       #what CAMB stuff goes up to (2 to 10000 ignoring 10100)
lmax2=105000     #a bit over what we need for ACT resolution (33000ish)

def spectra():

  data_scal=np.loadtxt(working_dir+'/home/zahra/python_scripts/CMB_noise/new_scalCls.dat')
  data_lens=np.loadtxt(working_dir+'/home/zahra/python_scripts/CMB_noise/new_lensedCls.dat')

  data_scal=data_scal[0:lmax,:]
  data_lens=data_lens[0:lmax,:]
  pp_factor=7.4311e+12  # factor that normalises large-scale structure relative to cmb, (10^6 *Tcmb)^2
  l_vec=data_scal[:,0]
  l_arr_sq = l_vec*l_vec
  rescale = l_vec * (l_vec + 1.0)/(2*pi)   #Jet commented out division by 2pi, I put it back after reading camb readme

  cl_tt=(data_scal[:,1]/(rescale))
  cl_ee=(data_scal[:,2]/(rescale))
  cl_te=(data_scal[:,3]/(rescale))

  cl_tt_lens=data_lens[:,1]/(rescale)
  cl_ee_lens=data_lens[:,2]/(rescale)
  cl_bb_lens=data_lens[:,3]/(rescale)
  cl_te_lens=data_lens[:,4]/(rescale)



  cl_dd=(data_scal[:,4]/l_arr_sq)/pp_factor
  cl_pp=cl_dd/l_arr_sq   #made this agree with CAMB readme by replacing rescale with l_arr_sq
  cl_kk=(rescale*2*pi)**2*cl_pp/4 #after putting back the /2pi in rescale, I cancelled it out here by multiplying by 2pi





  #making longer so that when we need the spectrum for large l we can give a sensble answer
  l_vec_extra=np.arange(l_vec[l_vec.size-1]+1,l_vec[l_vec.size-1]+1+(lmax2-lmax), 1) #need to go out further for higher resolution
  l_vec_long=np.concatenate((l_vec,l_vec_extra))

  cl_tt_lens_extra=(l_vec_extra**(-6))
  cl_tt_lens_extra=cl_tt_lens_extra*cl_tt_lens[l_vec.size-1]/cl_tt_lens_extra[0]
  cl_tt_lens_long=np.concatenate((cl_tt_lens,cl_tt_lens_extra))

  cl_tt_extra=(l_vec_extra**(-5))
  cl_tt_extra=cl_tt_extra*cl_tt[l_vec.size-1]/cl_tt_extra[0]
  cl_tt_long=np.concatenate((cl_tt,cl_tt_extra))

  cl_ee_lens_extra=(l_vec_extra**(-6))
  cl_ee_lens_extra=cl_ee_lens_extra*cl_ee_lens[l_vec.size-1]/cl_ee_lens_extra[0]
  cl_ee_lens_long=np.concatenate((cl_ee_lens,cl_ee_lens_extra))

  cl_ee_extra=(l_vec_extra**(-5.5))
  cl_ee_extra=cl_ee_extra*cl_ee[l_vec.size-1]/cl_ee_extra[0]
  cl_ee_long=np.concatenate((cl_ee,cl_ee_extra))

  cl_te_lens_extra=(l_vec_extra**(-6))
  cl_te_lens_extra=cl_te_lens_extra*cl_te_lens[l_vec.size-1]/cl_te_lens_extra[0]
  cl_te_lens_long=np.concatenate((cl_te_lens,cl_te_lens_extra))

  cl_te_extra=(l_vec_extra**(-5.5))
  cl_te_extra=cl_te_extra*cl_te[l_vec.size-1]/cl_te_extra[0]
  cl_te_long=np.concatenate((cl_te,cl_te_extra))

  cl_bb_lens_extra=(l_vec_extra**(-6))
  cl_bb_lens_extra=cl_bb_lens_extra*cl_bb_lens[l_vec.size-1]/cl_bb_lens_extra[0]
  cl_bb_lens_long=np.concatenate((cl_bb_lens,cl_bb_lens_extra))

  cl_kk_extra=(l_vec_extra**(-2.8))
  cl_kk_extra=cl_kk_extra*cl_tt[l_vec.size-1]/cl_kk_extra[0]
  cl_kk_long=np.concatenate((cl_kk,cl_kk_extra))
  """
  plt.plot(l_vec_long, cl_bb_lens_long)
  plt.yscale('log')
  plt.show()
  """
  #return(l_vec,cl_tt, cl_ee, cl_te, cl_tt_lens, cl_ee_lens, cl_bb_lens, cl_te_lens)
  return(l_vec_long,cl_tt_long, cl_ee_long, cl_te_long, cl_tt_lens_long, cl_ee_lens_long, cl_bb_lens_long, cl_te_lens_long, cl_kk_long)


l_vec,cl_tt, cl_ee, cl_te, cl_tt_lens, cl_ee_lens, cl_bb_lens, cl_te_lens, cl_kk=spectra()

#to get deflection spec without messing with other code
cl_dd=cl_kk* 4/ (l_vec * (l_vec + 1.0))
cl_pp=cl_dd/(l_vec * (l_vec + 1.0))

rms_tt=np.sqrt(cl_tt)
rms_e1=cl_te/rms_tt

cl_e2=cl_ee-(cl_te**2/cl_tt)
rms_e2=np.sqrt(cl_e2)

cl_tt_func=spline.Spline(l_vec,cl_tt)
cl_ee_func=spline.Spline(l_vec,cl_ee)
cl_te_func=spline.Spline(l_vec,cl_te)

cl_tt_lens_func=spline.Spline(l_vec,cl_tt_lens)
cl_ee_lens_func=spline.Spline(l_vec,cl_ee_lens)
cl_te_lens_func=spline.Spline(l_vec,cl_te_lens)

cl_kk_func=spline.Spline(l_vec,cl_kk)
cl_dd_func=spline.Spline(l_vec,cl_dd)
cl_pp_func=spline.Spline(l_vec,cl_pp)

rms_tt_func=spline.Spline(l_vec,rms_tt)
rms_e1_func=spline.Spline(l_vec,rms_e1)
rms_e2_func=spline.Spline(l_vec,rms_e2)

hsqrt2=0.5*sqrt(2)


def getNoiseTT(ls, expt):
    if expt=='planck':
        freq_channels=np.array([100.0, 143.0, 217.0])#  #GHz
        theta_fwhm=np.array([0.18, 0.13, 0.092])*60#         #arcmin
        sigma_p=np.array([4.5,  5.5,  11.8])#                #detector noise in pixel of side thetaBeam, 1 micro kelvin
    elif expt=='act':
        freq_channels=np.array([145.0, 220.0, 265.0]) #([100.0, 143.0, 217.0])#  #GHz
        theta_fwhm=np.array([1.7, 1.1, 0.9])   #([0.18, 0.13, 0.092])*60#         #arcmin
        sigma_p=np.array([2.0, 5.2, 8.8])    #([4.5,  5.5,  11.8])#                #detector noise in pixel of side thetaBeam, 1 micro kelvin (per arcminute???)

    elif expt=='ref':
        freq_channels=np.array([150.0]) #([100.0, 143.0, 217.0])#  #GHz
        theta_fwhm=np.array([1.5])   #([0.18, 0.13, 0.092])*60#         #arcmin
        sigma_p=np.array([1.5])    #([4.5,  5.5,  11.8])#                #detector noise in pixel of side thetaBeam, 1 micro kelvin (per arcminute???)

    elif expt=='advact':    #just 150 GHz to match kernels
        freq_channels=np.array([150.])   #GHz
        theta_fwhm=np.array([1.4])         #arcmin
        sigma_p=np.array([7.]) #microK arcmin -->need to convert to noise in beam!!

    elif expt =='pix':
        freq_channels=np.array([150.0])
        sigma_p=np.array([1.0]) # uK arcmin
        theta_fwhm = np.array([1.0]) # arcmin

    theta_beam=theta_fwhm*np.pi/(180*60)        #radians

    b_sq=np.zeros(ls.shape[0], dtype=np.float64)
    for i in range(freq_channels.shape[0]):      #try do without loop
        b_sq+=1/(theta_beam[i]*sigma_p[i])**2*np.exp((-ls*(ls+np.ones(ls.shape[0]))*theta_beam[i]**2)/(8*np.log(2)))
    #print b_sq
    nl=np.zeros(ls.shape[0], dtype=np.float64)
    nl=1/b_sq
    #print nl
    return (nl)

nl_tt_planck=np.nan_to_num(getNoiseTT(l_vec, 'planck'))
nl_tt_act=np.nan_to_num(getNoiseTT(l_vec, 'act'))
#Zahra's edition-advact(using ref which is more accurate)
nl_tt_ref=np.nan_to_num(getNoiseTT(l_vec, 'ref'))


nl_tt_planck_func=spline.Spline(l_vec, nl_tt_planck)
nl_tt_act_func=spline.Spline(l_vec, nl_tt_act)
nl_tt_ref_func=spline.Spline(l_vec, nl_tt_ref)


def nl_tt_planck_func_bis(l):
    return nl_tt_planck_func(l)

def nl_tt_act_func_bis(l):
    return nl_tt_act_func(l)

def nl_tt_ref_func_bis(l):
    return nl_tt_ref_func(l)

#Zahra's comments
l_vec=l_vec[:len(l_vec)-90000]
#l_vec=l_vec[:len(l_vec)-10000]

#print (l_vec)
nl_tt_act_shaped=nl_tt_act_func_bis(l_vec)


#print (nl_tt_act_shaped.shape)
#End of Zahra's additions

def cl_pp_func_bis(l):
    try:
        return (cl_pp_func(l))
    except spline.SplineRangeError:
        return(0.)

def cl_dd_func_bis(l):
    try:
        return (cl_dd_func(l))
    except spline.SplineRangeError:
        return(0.)

def cl_kk_func_bis(l):
  try:
    return (cl_kk_func(l))
  except spline.SplineRangeError:
    return(0.)

def tt_func_bis(l):
  try:
    return (rms_tt_func(l))
  except spline.SplineRangeError:
    return(0.)

def e1_func_bis(l):
  try:
    return (rms_e1_func(l))
  except spline.SplineRangeError:
    return(0.)

def e2_func_bis(l):
  try:
    return (rms_e2_func(l))
  except spline.SplineRangeError:
    return(0.)

def cl_tt_func_bis(l):
  try:
    return (cl_tt_func(l))
  except spline.SplineRangeError:
    return(0.)

#Zahra's comments
#I added the plots
OV_full_signal=lpkSZ.interp_OV_full_signal
'''
def quad_addition(func):
    squares=np.array([])
    for i in func:
        square_ind=i**2
        squares=np.append(squares,square_ind)
    sum=np.sum(squares)
    sqrt=np.sqrt(sum)
    return sqrt

OV_summed=quad_addition(OV_full_signal)
'''

def SNR_Integrand(ell): #I'm not sure of this. Nmodes=2*l*sky*delta_l. When integrating, the delta l becomes dl and the l is part of the
#integrand. But without integrating, the l is probably the annulus centre with the delta l being the radius of the annulus. I'm not sure.
#I decided not to plot the integrand
    f_sky=0.36
    ell_c=2500.
    delta_ell=2500.
    const=2.*f_sky/2./np.pi
    Cl=OV_full_signal(ell)
    Nl=cl_tt_func_bis(ell)+nl_tt_ref_func_bis(ell)
    integrand=ell*(Cl/np.sqrt(2.)/(Cl+Nl))**2
    return np.sqrt(const*integrand)
'''
plt.xlabel('l')
plt.ylabel('Differential S/N')
plt.plot(l_vec,SNR_Integrand(l_vec))
plt.show()
'''
def SNR(ell):
    f_sky=0.36
    const=2.*f_sky/(2.*np.pi)
    Cl=OV_full_signal(ell)
    Nl=cl_tt_func_bis(ell)+nl_tt_ref_func_bis(ell)
    integrand=const*ell*(Cl/(Cl+Nl))**2/2.
    integral=sp.integrate.cumtrapz(integrand,ell,initial=0)
    return np.sqrt(integral)
#print (OV_summed)
#The answer is: 0.0023310954554638607
'''
plt.xlabel('l')
#plt.ylabel(r'$\rm{l(l+1)C_l/(2 \pi)[\mu K^2]}$')
plt.ylabel('Cumulative S/N')
plt.plot(l_vec,SNR(l_vec))
#plt.ylim(0,100)

plt.loglog(l_vec,l_vec*(l_vec+1)*OV_full_signal(l_vec)/2/np.pi,'g')
plt.loglog(l_vec,l_vec*(l_vec+1)*cl_tt_func_bis(l_vec)/2/np.pi,'r')
#plt.loglog(l_vec,OV_full_signal/(nl_tt_ref_shaped+cl_tt_func_bis(l_vec)))
plt.loglog(l_vec,l_vec*(l_vec+1)*nl_tt_ref_func_bis(l_vec)/2/np.pi,'b')
#plt.loglog(l_vec,nl_tt_act_shaped/2/np.pi,'r')
#plt.legend(labels=('AdvACT','ACT'))
plt.legend(labels=('OV','cl_tt','nl_tt'))
'''



#End of Zahra's comments

def cl_tt_func_bis_lots(ls):
  try:
    result=np.zeros(ls.shape)
    for i, l in enumerate(ls):
        result[i]=cl_tt_func(l)
    return result
  except spline.SplineRangeError:
    return(0.)


def cl_tt_func_bis_matrix(ls):
  try:
    result=np.zeros(ls.shape)
    for i, arr in enumerate(ls):
        for j, l in enumerate(arr):
            result[i, j]=cl_tt_func(l)
    return result
  except spline.SplineRangeError:
    return(0.)


def cl_ee_func_bis(l):
  try:
    return (cl_ee_func(l))
  except spline.SplineRangeError:
    return(0.)

def cl_te_func_bis(l):
  try:
    return (cl_te_func(l))
  except spline.SplineRangeError:
    return(0.)

def cl_tt_lens_func_bis(l):
  try:
    return (cl_tt_lens_func(l))
  except spline.SplineRangeError:
    return(0.)

def cl_tt_lens_func_bis_lots(ls):
  try:
    result=np.zeros(ls.shape)
    for i, l in enumerate(ls):
        result[i]=cl_tt_lens_func(l)
    return result
  except spline.SplineRangeError:
    return(0.)

def cl_tt_lens_func_bis_matrix(ls):
  try:
    result=np.zeros(ls.shape)
    for i, arr in enumerate(ls):
        for j, l in enumerate(arr):
            result[i, j]=cl_tt_lens_func(l)
    return result
  except spline.SplineRangeError:
    return(0.)

def cl_ee_lens_func_bis(l):
  try:
    return (cl_ee_lens_func(l))
  except spline.SplineRangeError:
    return(0.)

def cl_te_lens_func_bis(l):
  try:
    return (cl_te_lens_func(l))
  except spline.SplineRangeError:
    return(0.)


#power spectra from maps, Jet/Martin's code
#returns l**2*c(l) - this can cause confusion!!
def compute_power_spectrum(my_map,fov,sm_width=500):
   nside, trash=my_map.shape
   a_l=np.fft.fft2(my_map)
   c_l=abs(a_l)**2#c_l=abs(a_l**2) Heather changed
   ##print 'c_l', c_l
   l_arr=map_making_functions.make_l_array(nside,fov)
   ll_c_l=c_l*l_arr*l_arr
   ll_c_l.shape=(nside**2,)
   l_arr.shape =(nside**2,)
   ind_list=l_arr.argsort()
   l_arr=l_arr[ind_list]
   ll_c_l=ll_c_l[ind_list]
   ll_c_l_sm=smooth.smooth(ll_c_l,sm_width,"flat")

   result=(l_arr,ll_c_l_sm/(nside**4)) #?? change back???
   return(result)

#Jet/Martin's code
#returns l**2*c(l) - this can cause confusion!!
def compute_cross_spectrum(my_map_1,my_map_2,fov,sm_width=500):
   nside, trash=my_map_1.shape
   a_l_1=np.fft.fft2(my_map_1)
   a_l_2=np.fft.fft2(my_map_2)
   #c_l=abs(a_l_1*a_l_2) #make this complex conjugate
   c_l=np.conj(a_l_1)*a_l_2
   ##print 'power spectrum (should be real)',c_l
   c_l=abs(c_l)
   l_arr=map_making_functions.make_l_array(nside,fov)
   ll_c_l=c_l*l_arr*l_arr
   ll_c_l.shape=(nside**2,)
   l_arr.shape =(nside**2,)
   ind_list=l_arr.argsort()
   l_arr=l_arr[ind_list]
   ll_c_l=ll_c_l[ind_list]
   ll_c_l_sm=smooth.smooth(ll_c_l,sm_width,"flat")

   result=(l_arr,ll_c_l_sm/(nside**4))  #?? change back???
   return(result)

def compute_2d_power_spectrum(my_map,fov):
    nside, trash=my_map.shape
    fov_rad=fov*pi/180
    a_l=np.fft.fft2(my_map)*(fov_rad/nside)**2
    c_l=abs(a_l)**2
    l_arr=map_making_functions.make_l_array(nside,fov)
    return (l_arr, c_l)


#given 2 normalised fields in fourier space, finds binned power spec
#check error bit!
def get_c_l_with_error(a_l_1, a_l_2, fov_rad, nside, binsize, shear=False, spec=False):
    fov=fov_rad*180/pi
    c_l=np.conj(a_l_1)*a_l_2
    ##print 'power spectrum (should be real)',c_l
    c_l=abs(c_l)
    l_arr=map_making_functions.make_l_array(nside,fov)

    plt.imshow(c_l)

    ##print spec
    ##print shear

    lx_array, ly_array=map_making_functions.make_lxy_arrays(nside,fov)
    cos_2theta_array=(lx_array**2-ly_array**2)/(lx_array**2+ly_array**2)
    sin_2theta_array=(2*lx_array*ly_array)/(lx_array**2+ly_array**2)

    if shear=='shear_plus' and (spec=='tb' or spec=='eb'):
        #print 'tb/eb shear plus'
        c_for_err=c_l/sin_2theta_array**2
        fac=sin_2theta_array**2
    elif shear=='shear_cross'and (spec=='tb' or spec=='eb'):
        c_for_err=c_l/cos_2theta_array**2
        fac=cos_2theta_array**2
    elif shear=='shear_plus':
        c_for_err=c_l/cos_2theta_array**2
        fac=cos_2theta_array**2
    elif shear=='shear_cross':
        c_for_err=c_l/sin_2theta_array**2
        fac=sin_2theta_array**2
    """
    plt.imshow(np.fft.fftshift(l_arr))
    plt.colorbar()
    plt.show()

    plt.title(spec+' '+shear)
    plt.subplot(131)
    plt.imshow(np.fft.fftshift(l_arr*(l_arr+1)*c_l))
    plt.colorbar(shrink=0.6)
    plt.title('2D spectrum')
    plt.subplot(132)
    plt.imshow(np.fft.fftshift(fac))
    plt.colorbar(shrink=0.6)
    plt.title('cos/sin')
    plt.subplot(133)
    plt.imshow(np.isfinite(np.fft.fftshift((c_for_err))))#, vmin=0, vmax=4)
    plt.colorbar(shrink=0.6)
    plt.title('2D spec to use for error')
    plt.show()
    """
    """
    c_l=np.reshape(c_l,newshape=(nside**2,))
    c_for_err=np.reshape(c_for_err,newshape=(nside**2,))
    l_arr.shape =(nside**2,)
    ind_list=l_arr.argsort()
    l_arr=l_arr[ind_list]
    c_l=c_l[ind_list]
    c_for_err=c_for_err[ind_list]
    """
    #c_l=smooth.smooth(c_l,500,"flat")

    ls=np.arange(3, lmax, binsize)
    c_l_binned=np.zeros(ls.shape)
    error=np.zeros(ls.shape)
    for i,l in enumerate(ls):
        c_l_binned[i]=np.nanmean(c_l[np.where((l_arr>l)&(l_arr<l+binsize)&(np.isfinite(c_l)))])
        """
        if i==5 or i==20:
            plt.subplot(131)
            plt.imshow(c_l)
            plt.colorbar()
            plt.subplot(132)
            plt.imshow(2*c_l_binned[i]*cos_2theta_array**2)
            plt.colorbar()
            plt.subplot(133)
            plt.imshow((c_l-2*c_l_binned[i]*cos_2theta_array**2))
            plt.colorbar()
            plt.show()
            #print 'max=',np.amax((c_l-2*c_l_binned[i]*cos_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))])
            #print 'min=', np.amin((c_l-2*c_l_binned[i]*cos_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))])
            #print 'mean=', np.mean((c_l-2*c_l_binned[i]*cos_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))])"""
        if shear=='shear_plus' and (spec=='tb' or spec=='eb'):
            error[i]=np.nanstd((c_l-2*c_l_binned[i]*sin_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))])  #&(np.isfinite(c_for_err))
            if False:#l<200:
                plt.hist((c_l-2*c_l_binned[i]*sin_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))], bins=31)
                plt.title('c-2*cbin*sin^2(2phi)')
                plt.show()


        elif shear=='shear_cross'and (spec=='tb' or spec=='eb'):
            error[i]=np.nanstd((c_l-2*c_l_binned[i]*cos_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))])
            if False:
                plt.hist((c_l-2*c_l_binned[i]*cos_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))], bins=101)
                plt.title('c-2*cbin*cos^2(2phi)')
                plt.show()
        elif shear=='shear_plus':
            error[i]=np.nanstd((c_l-2*c_l_binned[i]*cos_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))])
            if False:
                plt.hist((c_l-2*c_l_binned[i]*cos_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))], bins=101)
                plt.title('c-2*cbin*cos^2(2phi)')
                plt.show()
        elif shear=='shear_cross':
            error[i]=np.nanstd((c_l-2*c_l_binned[i]*sin_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))])
            if False:
                plt.hist((c_l-2*c_l_binned[i]*sin_2theta_array**2)[np.where((l_arr>l)&(l_arr<l+binsize))], bins=101)
                plt.title('c-2*cbin*sin^2(2phi)')
                plt.show()
        else:
            error[i]=np.nanstd(c_l[np.where((l_arr>l)&(l_arr<l+binsize))])
            if False:# l<200:
                plt.hist(c_l[np.where((l_arr>l)&(l_arr<l+binsize))], bins=31)
                plt.title('Pixel histogram for power in annulus for '+spec+' conv for l='+str(l))
                plt.show()

                plt.hist(np.absolute(a_l_1[np.where((l_arr>l)&(l_arr<l+binsize))]), bins=31)
                plt.title('Pixel histogram for amplitude in annulus for '+spec+' conv for l='+str(l))
                plt.show()


    ##print l, 'has', ((l_arr>l)&(l_arr<l+binsize)).sum(),'bins'
    ls+=binsize/2
    if shear=='shear_plus' or shear=='shear_cross':
        error/=2    #is this def legit?
    ##print c_l_binned.shape

    patch_to_sky_factor=2*pi*fov_rad**2/(4*pi)
    c_l_binned/=patch_to_sky_factor
    error/=patch_to_sky_factor

    #c_l_binned*=2*pi    #I think this has to go here because of my Fourier conventions, but it is only needed for reconstructed spectra??
    result=(ls,c_l_binned, error)    #multiply by 2pi**2 coz using Hu&Okamoto's Fourier conventions??
    return(result)



#just cl, not l^2*cl, and not smoothed, Heather's version of power spec fn
def just_compute_power_spectrum(my_map,fov,binsize=10, shear=False, spec=False):
    result=just_compute_cross_spectrum(my_map, my_map, fov, binsize, shear, spec)
    return(result)

#just cl, not l^2*cl, and not smoothed, Heather's version od power spec fn
def just_compute_cross_spectrum(my_map_1,my_map_2,fov,binsize=10, shear=False, spec=False):
    lmax=10000
    nside=my_map_1.shape[0]
    fov_rad=fov*pi/180
    a_l_1=np.fft.fft2(my_map_1)*(fov_rad/nside)**2
    a_l_2=np.fft.fft2(my_map_2)*(fov_rad/nside)**2
    #c_l=abs(a_l_1*a_l_2) #make this complex conjugate

    l, cl, noise=get_c_l_with_error(a_l_1, a_l_2, fov_rad, nside, binsize, shear, spec)
    return(l,cl)

def compute_power_spectrum_with_error(my_map,fov,binsize=10, shear=False, spec=False):
    result=compute_cross_spectrum_with_error(my_map, my_map, fov, binsize, shear, spec)
    return(result)

def compute_cross_spectrum_with_error(my_map_1,my_map_2,fov,binsize=10, shear=False, spec=False):
    lmax=10000
    nside=my_map_1.shape[0]
    fov_rad=fov*pi/180
    a_l_1=np.fft.fft2(my_map_1)*(fov_rad/nside)**2
    a_l_2=np.fft.fft2(my_map_2)*(fov_rad/nside)**2
    #c_l=abs(a_l_1*a_l_2) #make this complex conjugate

    result=get_c_l_with_error(a_l_1, a_l_2, fov_rad, nside, binsize, shear, spec)
    return(result)

def compute_shear_EB_spec_from_plus_cross(plus_map, cross_map, fov,spec='EE',binsize=10):
    nside=plus_map.shape[0]
    fov_rad=fov*pi/180

    f_plus=np.fft.fft2(plus_map)
    f_cross=np.fft.fft2(cross_map)
    lx_array, ly_array=map_making_functions.make_lxy_arrays(nside,fov)  #fov in degrees
    cos_2phi_l=(lx_array**2-ly_array**2)/(lx_array**2+ly_array**2)
    sin_2phi_l=(2*lx_array*ly_array)/(lx_array**2+ly_array**2)
    cos_2phi_l[0,0]=0#1
    sin_2phi_l[0,0]=0#1

    f_E=cos_2phi_l*f_plus+sin_2phi_l*f_cross
    f_B=-1*sin_2phi_l*f_plus+cos_2phi_l*f_cross



    if spec=='EE':
        a_l_1=a_l_2=f_E*(fov_rad/nside)**2
    elif spec=='EB':
        a_l_1=f_E*(fov_rad/nside)**2
        a_l_2=f_B*(fov_rad/nside)**2
    elif spec=='BB':
        a_l_1=a_l_2=f_B*(fov_rad/nside)**2


    result=get_c_l(a_l_1, a_l_2, fov_rad, nside, binsize)
    return(result)




def compute_deflection_spectrum(alpha_x,alpha_y,fov,binsize=20):
    nside=alpha_x.shape[0]
    fov_rad=fov*pi/180

    a_l_x=np.fft.fft2(alpha_x)*(fov_rad/nside)**2
    a_l_y=np.fft.fft2(alpha_y)*(fov_rad/nside)**2

    c_l_xy=np.abs(np.conj(a_l_x)*a_l_y)
    c_l_x=np.abs(a_l_x)**2
    c_l_y=np.abs(a_l_y)**2

    l_arr=map_making_functions.make_l_array(nside,fov)
    lx,ly=map_making_functions.make_lxy_arrays(nside,fov)

    c_l_alpha=1/l_arr**2*(lx**2*c_l_x + ly**2*c_l_y)# + 2*lx*ly*c_l_xy)
    c_l_x_rescaled=lx**2*c_l_x/l_arr**2
    c_l_y_rescaled=ly**2*c_l_y/l_arr**2
    c_l_xy_rescaled=2*lx*ly*c_l_xy/l_arr**2

    ##print 'alpha x spec:', c_l_x
    ##print 'alpha spec:', c_l_alpha

    ls=np.arange(3, lmax, binsize)
    c_l_binned=np.zeros(ls.shape)
    c_l_x_binned=np.zeros(ls.shape)
    c_l_y_binned=np.zeros(ls.shape)
    c_l_xy_binned=np.zeros(ls.shape)
    for i,l in enumerate(ls):
        c_l_binned[i]=np.average(c_l_alpha[np.where((l_arr>l)&(l_arr<l+binsize))])
        c_l_x_binned[i]=np.average(c_l_x_rescaled[np.where((l_arr>l)&(l_arr<l+binsize))])
        c_l_y_binned[i]=np.average(c_l_y_rescaled[np.where((l_arr>l)&(l_arr<l+binsize))])
        c_l_xy_binned[i]=np.average(c_l_xy_rescaled[np.where((l_arr>l)&(l_arr<l+binsize))])
        ##print l, 'has', ((l_arr>l)&(l_arr<l+binsize)).sum(),'bins'
    ls+=binsize/2

    patch_to_sky_factor=2*pi*fov_rad**2/(4*pi)
    c_l_binned/=patch_to_sky_factor
    c_l_x_binned/=patch_to_sky_factor
    c_l_y_binned/=patch_to_sky_factor
    c_l_xy_binned/=patch_to_sky_factor


    def_mag=np.sqrt(alpha_x**2+alpha_y**2)
    c_l_mag=np.abs(np.fft.fft2(def_mag)*(fov_rad/nside)**2)**2
    c_l_mag_binned=np.zeros(ls.shape)
    for i,l in enumerate(ls):
        c_l_mag_binned[i]=np.average(c_l_mag[np.where((l_arr>l)&(l_arr<l+binsize))])
    c_l_mag_binned/=patch_to_sky_factor


    plt.plot(ls, np.sqrt(ls*(ls+1)/(2*pi)*c_l_binned), label='x and y combined from map')
    #plt.plot(ls, np.sqrt(ls*(ls+1)/(2*pi)*c_l_mag_binned), label='from map')
    #plt.plot(ls, np.sqrt(ls*(ls+1)/(2*pi)*c_l_x_binned), label=r'$\left(\frac{l_x}{l}\right)^2 c_l^{\alpha_x}$ from map')
    #plt.plot(ls, np.sqrt(ls*(ls+1)/(2*pi)*c_l_y_binned), label=r'$\left(\frac{l_x}{l}\right)^2 c_l^{\alpha_y}$ from map')
    #plt.plot(ls, np.sqrt(ls*(ls+1)/(2*pi)*np.abs(c_l_xy_binned)), label=r'$2\frac{l_x l_y}{l^2} c_l^{\alpha_x\alpha_y}$ from map')
    plt.plot(l_vec, np.sqrt(l_vec*(l_vec+1)/(2*pi)*cl_dd), label='from camb')
    plt.legend(loc='lower center')
    plt.xlim(1,10000)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('L')
    plt.ylabel(r'$[l(l+1)C^{\alpha\alpha}_L/2\pi]^{1/2}$')
    plt.title('deflection angle power spec')
    plt.show()

    result=(ls,c_l_binned)
    return(result)
