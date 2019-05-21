import numpy as np
from math import *
import matplotlib.pyplot as plt
import scal_power_spectra
import map_making_functions
import sys
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d

args=sys.argv
nside=int(args[1])     
fov_deg=float(args[2])
num_maps=20#[1, 20]#,10,20]#, 20, 50]#,20,40,60]#, 50]
num_rand_phase_maps=0#20
spec='EB'
est='shear_plus'

exp='ref'#'planck'#'ref' or 'planck', add 'act'
#working_dir='/Users/Heather/Dropbox/PolRealSp/Codes/MapsPyJet/HeatherVersion'
#saving_dir='/Users/Heather/Documents/HeatherVersionPlots/planck'
if(exp=='ref' or exp=='planck'):
    working_dir=saving_dir='/Users/Heather/Documents/HeatherVersionPlots/'+exp+'/fov'+str(int(fov_deg))#+'/jetnormconv'
elif (exp[0:3]=='ref'):
    working_dir=saving_dir='/Users/Heather/Documents/HeatherVersionPlots/ref/fov'+str(int(fov_deg))
datafile_saving_dir='/Users/Heather/Documents/HeatherVersionPlots/planck/fov'+str(int(fov_deg))+'/Datafiles'
N0_directory='/Users/Heather/Dropbox/PolRealSp/Codes/HeatherVersion/NoisePlots/DataFiles'
HS_dir='/Users/Heather/Dropbox/PolRealSp/Codes/HeatherVersion/HarmonicSpace'

power_spectrum=scal_power_spectra.just_compute_power_spectrum
cross_spectrum=scal_power_spectra.just_compute_cross_spectrum

subtract_N0=False   #make that section work better!
use_noisy_maps=False

fudge=1.0   #this is part of Jethro's map filenames, I need to make new naming convention and then remove this
i=0         #this is part of Jethro's map filenames, I need to make new naming convention and then remove this
lmax=9999       #what CAMB stuff goes up to (2 to 10000 ignoring 10100)
A=(fov_deg*(pi/180.))**2

pp_factor=7.4311e+12    #experiments with normalisation
print 'Spectrum:', spec
#theoretical spectra from CAMB
l_vec,cl_tt, cl_ee, cl_te, cl_tt_lens, cl_ee_lens, cl_bb_lens, cl_te_lens, cl_kk=scal_power_spectra.spectra()

cl_dd=cl_kk*4 / (l_vec*(l_vec+1))

t_cmb=2.728
if exp=='planck':
    sigma_am=7.
    delta_t_am=27.
    delta_p_am=40*sqrt(2)

if exp=='ref':
    sigma_am=1.#60*fov_deg/nside
    delta_t_am=1.
    delta_p_am=delta_t_am*sqrt(2)
elif (exp[0:3]=='ref'):
    delta_t_am=float(exp[3:])
    delta_p_am=delta_t_am*sqrt(2)
    sigma_am=1.

sigma_rad=(sigma_am)*(pi/(60*180))
delta_t=delta_t_am*(pi/(60*180))
delta_p=delta_p_am*(pi/(60*180))

n_l_tt=(delta_t/t_cmb)**2*np.exp(l_vec*(l_vec+1)*sigma_rad**2/(8*log(2)))
n_l_ee=(delta_p/t_cmb)**2*np.exp(l_vec*(l_vec+1)*sigma_rad**2/(8*log(2)))
n_l_bb=n_l_ee
"""#leave commented
if spec=='tt':
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_tt[0:lmax], label=r'$l(l+1)c^{TT}_l$')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_tt_lens[0:lmax], label=r'$l(l+1)\tilde{c}^{TT}_l$')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*n_l_tt[0:lmax], label=r'$l(l+1)n^{TT}_l$')
    plt.legend()
    plt.yscale('log')
    plt.title('Temperature Power Spectra from CAMB')
    plt.savefig(saving_dir+'/Spectra/TT_spectra_camb_beam'+str(round(sigma_am, 2))+'am_noise'+str(delta_t_am)+'.png')
    plt.show()
    
    
if spec=='ee':
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_ee[0:lmax], label=r'$l(l+1)c^{EE}_l$')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_ee_lens[0:lmax], label=r'$l(l+1)\tilde{c}^{EE}_l$')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*n_l_ee[0:lmax], label=r'$l(l+1)n^{EE}_l$')
    plt.legend()
    plt.yscale('log')
    plt.title('E Mode Polarisation Power Spectra from CAMB')
    plt.savefig(saving_dir+'/Spectra/EE_spectra_camb_beam'+str(round(sigma_am, 2))+'am_noise'+str(round(delta_t_am*sqrt(2), 2))+'.png')
    plt.show()
    
    
if spec=='te':
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*np.absolute(cl_te[0:lmax]), label=r'$l(l+1)c^{TE}_l$')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*np.absolute(cl_te_lens[0:lmax]), label=r'$l(l+1)\tilde{c}^{TE}_l$')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*n_l_ee[0:lmax], label=r'$l(l+1)n^{EE}_l$')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*n_l_tt[0:lmax], label=r'$l(l+1)n^{TT}_l$')
    plt.legend()
    plt.yscale('log')
    plt.title('Temperature and E Mode Polarisation Cross Power Spectra from CAMB')
    plt.savefig(saving_dir+'/Spectra/TE_spectra_camb_beam'+str(round(sigma_am, 2))+'am_noiseT'+str(delta_t_am)+'.png')
    plt.show()
    
if spec=='bb':
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_bb_lens[0:lmax], label=r'$l(l+1)\tilde{c}^{BB}_l$')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*n_l_bb[0:lmax], label=r'$l(l+1)n^{BB}_l$')
    plt.legend()
    plt.yscale('log')
    plt.title('B Mode Polarisation Power Spectra from CAMB')
    plt.savefig(saving_dir+'/Spectra/BB_spectra_camb_beam'+str(round(sigma_am, 2))+'am_noise'+str(round(delta_t_am*sqrt(2), 2))+'.png')
    plt.show()
"""
    
#uncomment below to plot map spectra in addition to convergence spectra

T=np.loadtxt(working_dir+'/Maps/T_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))
E=np.loadtxt(working_dir+'/Maps/E_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))
B=np.loadtxt(working_dir+'/Maps/B_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))

T_lensed=np.loadtxt(working_dir+'/Maps/T_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
E_lensed=np.loadtxt(working_dir+'/Maps/E_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
B_lensed=np.loadtxt(working_dir+'/Maps/B_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
if(use_noisy_maps):
    just_noise_T=np.loadtxt(working_dir+'/Maps/just_noise_T_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
    just_noise_E=np.loadtxt(working_dir+'/Maps/just_noise_E_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
    just_noise_B=np.loadtxt(working_dir+'/Maps/just_noise_B_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
else:
    just_noise_T=np.zeros((nside,nside))
    just_noise_E=np.zeros((nside,nside))
    just_noise_B=np.zeros((nside,nside))


#l_arr, n_l_tt_map=power_spectrum(just_noise_T,fov_deg)
#l_arr, n_l_ee_map=power_spectrum(just_noise_E,fov_deg)
l_arr, n_l_bb_map=power_spectrum(just_noise_B,fov_deg)

l_arr_fac=l_arr*(l_arr+np.ones(l_arr.shape))/(2*pi)
l_vec_fac=l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))/(2*pi)



def make_spec_plots(XY, map1, map2, map1lens, map2lens, l_vec, cl_xy, cl_xy_lens):
    pass
    """l_arr, cl_xy_map=cross_spectrum(map1, map2, fov_deg)
    l_arr,cl_xy_lens_map=cross_spectrum(map1lens, map2lens,fov_deg)

    #plot lensed and unlensed power spectra
    plt.plot(l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))/(2*pi)*np.abs(cl_xy[0:lmax]), label=r'$l(l+1)c_l/2\pi$ camb')
    plt.plot(l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))/(2*pi)*np.abs(cl_xy_lens[0:lmax]), label=r'$l(l+1)\tilde{c}_l/2\pi$ camb')
    plt.plot(l_arr, l_arr*(l_arr+np.ones(l_arr.shape))/(2*pi)*np.abs(cl_xy_map), label=r'$l(l+1)c_l/2\pi$ map')
    plt.plot(l_arr, l_arr*(l_arr+np.ones(l_arr.shape))/(2*pi)*np.abs(cl_xy_lens_map), label=r'$l(l+1)\tilde{c}_l/2\pi$ map')
    plt.legend(loc='lower center', ncol=2) #, bbox_to_anchor=(0.5, -0.005)
    plt.yscale('log')
    plt.xscale('log')
    #plt.ylim(1, 1e4)
    plt.title(XY+' Power Spectra')
    plt.savefig(saving_dir+'/Spectra/'+XY+'_spectra_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+'_loglog.png')
    plt.show()
    plt.close()
    
    print cl_xy_map/cl_xy[l_arr]
    plt.plot(l_arr, cl_xy_map/cl_xy[l_arr])
    plt.show()
    
    if(True):#XY=='TT' or XY=='EE' or XY=='TE'):
        #plot fractional difference between lensed and unlensed power spectra
        plt.plot(l_arr, (cl_xy_lens_map-cl_xy_map)/cl_xy_map, label=r'$(\tilde{c}_l-c_l)/c_l$ map')
        plt.plot(l_vec[0:lmax], (cl_xy_lens[0:lmax]-cl_xy[0:lmax])/cl_xy[0:lmax],label=r'$(\tilde{c}_l-c_l)/c_l$ camb')
        plt.xlim(0, 3000)
        plt.ylim(-0.1, 0.25)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=2)
        plt.title('Fractional change in '+XY+' power spectrum due to lensing')
        plt.savefig(saving_dir+'/Spectra/'+XY+'_fractional_difference_due_to_lensing'+'_'+str(nside)+'_'+str(int(fov_deg))+'.png')
        plt.show()
        plt.close()
    
        #plot log difference between lensed and unlensed power spectra
        plt.plot(l_arr, l_arr_fac*np.abs((cl_xy_lens_map-cl_xy_map)), label=r'$l(l+1)(\tilde{c}_l-c_l)$ map')
        plt.plot(l_vec[0:lmax], l_vec_fac*np.abs(cl_xy_lens[0:lmax]-cl_xy[0:lmax]),label=r'$l(l+1)(\tilde{c}_l-c_l))$ camb')
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim(1e-3, 1e3)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=2)
        plt.title('Difference in power spectrum due to lensing')
        plt.savefig(saving_dir+'/Spectra/'+XY+'_log_difference_due_to_lensing'+'_'+str(nside)+'_'+str(int(fov_deg))+'.png')
        plt.show()
        plt.close()
    """
if spec=='tt':
    make_spec_plots('TT', T, T, T_lensed, T_lensed, l_vec, cl_tt, cl_tt_lens)
    """
    T_lensed=np.loadtxt(working_dir+'/Maps/T_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
    l_arr,cl_tt_lens_map=power_spectrum(T_lensed,fov_deg)
    diff=np.zeros(l_arr.shape)
    frac_diff_diff=np.zeros(l_arr.shape)
    max_num_maps=num_maps[len(num_maps)-1]
    for i in range(max_num_maps):
        T=np.loadtxt(working_dir+'/Maps/T_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))
        T_lensed=np.loadtxt(working_dir+'/Maps/T_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        l_arr,cl_tt_map=power_spectrum(T,fov_deg)
        l_arr,cl_tt_lens_map=power_spectrum(T_lensed,fov_deg)
        cl_tt_camb_spline=spline1d(l_vec, cl_tt)
        cl_tt_camb=cl_tt_camb_spline(l_arr)
        cl_tt_lens_camb_spline=spline1d(l_vec, cl_tt_lens)
        cl_tt_lens_camb=cl_tt_lens_camb_spline(l_arr)
    
        diff+=cl_tt_lens_camb-cl_tt_lens_map
        frac_diff_diff+=(cl_tt_lens_camb-cl_tt_camb)/cl_tt_camb-(cl_tt_lens_map-cl_tt_map)/cl_tt_map
        plt.plot(l_arr,frac_diff_diff)
        plt.xlim(0, 3000)
        plt.ylim(-0.05,0.05)
        plt.title(str(i))
        plt.show()
    diff/=max_num_maps
    frac_diff_diff/=max_num_maps
    plt.plot(l_arr, diff)
    plt.title('average difference')
    plt.xlim(0, 3000)
    plt.ylim(-0.05,0.05)
    plt.show()

    plt.plot(l_arr, frac_diff_diff)
    plt.plot(l_arr, np.zeros(l_arr.shape))
    plt.title('Average difference for 20 maps between camb and map fractional difference due to lensing')
    plt.xlim(0, 3000)
    plt.ylim(-0.05,0.05)
    plt.show()
    """


    """
    l_arr,cl_tt_map=power_spectrum(T,fov_deg)
    l_arr,cl_tt_lens_map=power_spectrum(T_lensed,fov_deg)
    
    
    rescaleTemp=1#/7.4311e+12#np.amax(l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_tt[0:lmax])/np.amax(cl_tt_map)
    rescaleTlensed=1#/7.4311e+12#np.amax(l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_tt_lens[0:lmax])/np.amax(cl_tt_lens_map)
    rescale=1#/(7.4311e+12*2*pi)
    rescaleTNoise=1#l_vec[526]**2*n_l_tt[526]/n_l_tt_map[1515]
    
    cib_poisson=l_vec**2/900000


    plt.plot(l_vec_fac*cl_tt[0:lmax]*rescale, label=r'$l(l+1)c^{TT}_l/2\pi$ camb')
    plt.plot(l_vec_fac*cl_tt_lens[0:lmax]*rescale, label=r'$l(l+1)\tilde{c}^{TT}_l/2\pi$ camb')
    
    
    #plt.plot(l_vec[0:lmax],cib_poisson[0:lmax], label='cib poisson') #*l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))/(2*pi))
    
    plt.plot(l_arr, l_arr_fac*cl_tt_map*rescaleTemp, label=r'$l(l+1)c^{TT}_l/2\pi$ map')
    plt.plot(l_arr, l_arr_fac*cl_tt_lens_map*rescaleTlensed, label=r'$l(l+1)\tilde{c}^{TT}_l/2\pi$ map')
    
    
    plt.plot(l_vec_fac*n_l_tt[0:lmax], label=r'$l(l+1)n^{TT}_l$ theory')
    #plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*n_l_tt_2[0:lmax], label=r'$l(l+1)n^{TT}_l$ theory 2')
    plt.plot(l_arr, l_arr_fac*n_l_tt_map*rescaleTNoise, label=r'$l(l+1)n^{TT}_l$ map')
    plt.legend(loc='lower center', ncol=3) #, bbox_to_anchor=(0.5, -0.005)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1, 1e4)
    plt.title('Temperature Power Spectra')
    #plt.savefig(saving_dir+'/Spectra/TT_spectra_beam'+str(round(sigma_am, 2))+'am_noise'+str(delta_t_am)+'.png')
    plt.savefig(saving_dir+'/Spectra/TT_spectra_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+'_loglog.png')
    plt.show()
    plt.close()

    

    plt.plot(l_arr, (cl_tt_lens_map-cl_tt_map)/cl_tt_map, label=r'$(\tilde{c}^{TT}_l-c^{TT}_l)/c^{TT}_l$ map')
    plt.plot(l_vec[0:lmax], (cl_tt_lens[0:lmax]-cl_tt[0:lmax])/cl_tt[0:lmax],label=r'$(\tilde{c}^{TT}_l-c^{TT}_l)/c^{TT}_l$ camb')
    plt.xlim(0, 3000)
    plt.ylim(-0.1, 0.25)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=2)
    plt.title('Fractional change in power spectrum due to lensing')
    plt.savefig(saving_dir+'/Spectra/tt_fractional_difference_due_to_lensing'+'_'+str(nside)+'_'+str(int(fov_deg))+'.png')
    plt.show()
    plt.close()

    plt.plot(l_arr, l_arr_fac*np.abs((cl_tt_lens_map-cl_tt_map)*rescaleTemp), label=r'$l(l+1)(\tilde{c}^{TT}_l-c^{TT}_l)$ map')
    print l_vec_fac.shape, l_vec[0:lmax].shape
    plt.plot(l_vec[0:lmax], l_vec_fac*np.abs(cl_tt_lens[0:lmax]-cl_tt[0:lmax]),label=r'$l(l+1)(\tilde{c}^{TT}_l-c^{TT}_l))$ camb')
    plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(1e-3, 1e3)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=2)
    plt.title('Difference in power spectrum due to lensing')
    plt.savefig(saving_dir+'/Spectra/tt_log_difference_due_to_lensing'+'_'+str(nside)+'_'+str(int(fov_deg))+'.png')
    plt.show()
    plt.close()

    plt.plot(l_arr, l_arr_fac*np.abs((cl_tt_lens_map-cl_tt_map)*rescaleTemp), label=r'$l(l+1)(\tilde{c}^{TT}_l-c^{TT}_l)/2\pi$ map')
    plt.plot(l_vec[0:lmax], l_vec_fac*np.abs(cl_tt_lens[0:lmax]-cl_tt[0:lmax]),label=r'$l(l+1)(\tilde{c}^{TT}_l-c^{TT}_l)/2\pi$ camb')
    plt.xlim(0, 3000)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.005), ncol=2)
    plt.title('Difference in power spectrum due to lensing')
    #plt.savefig(saving_dir+'/Spectra/tt_difference_due_to_lensing.png')
    plt.show()
    plt.close()
    """
                  
                  
                  

    """
    l_arr,cl_tt_map=power_spectrum(T,fov_deg)
    l_arr,cl_tt_lens_map=power_spectrum(T_lensed,fov_deg)

    
    plt.plot(l_arr, l_arr*(l_arr+1)*np.abs(cl_tt_lens_map-cl_tt_map)/(2*pi), label=r'$l(l+1)(\tilde{c}^{TT}_l-c^{TT}_l)/2\pi$ map')#*2*pi/(l_arr*(l_arr+1.))
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*np.abs(cl_tt_lens[0:lmax]-cl_tt[0:lmax])/(2*pi),label=r'$l(l+1)(\tilde{c}^{TT}_l-c^{TT}_l)/2\pi$ camb')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=2)
    plt.title('Difference in power spectrum due to lensing')
    plt.savefig(saving_dir+'/Spectra/tt_difference_due_to_lensing_averaged_spec.png')
    plt.show()
    plt.close()

    plt.plot(l_arr, l_arr*(l_arr+1)*(cl_tt_lens_map-cl_tt_map)/(2*pi), label=r'$l(l+1)(\tilde{c}^{TT}_l-c^{TT}_l)/2\pi$ map')#*2*pi/(l_arr*(l_arr+1.))
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*(cl_tt_lens[0:lmax]-cl_tt[0:lmax])/(2*pi),label=r'$l(l+1)(\tilde{c}^{TT}_l-c^{TT}_l)/2\pi$ camb')
    plt.xlim(0, 3000)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=2)
    plt.title('Difference in power spectrum due to lensing')
    #plt.savefig(saving_dir+'/Spectra/tt_difference_due_to_lensing.png')
    plt.show()
    plt.close()
    """

if spec=='ee':
    make_spec_plots('EE', E, E, E_lensed, E_lensed, l_vec, cl_ee, cl_ee_lens)
    
    """"
    l_arr,cl_ee_map=power_spectrum(E,fov_deg)
    l_arr,cl_ee_lens_map=power_spectrum(E_lensed,fov_deg)
    rescaleEE=2*pi#np.amax(l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_ee[0:lmax])/np.amax(cl_ee_map)
    rescaleEElensed=2*pi#p.amax(l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_ee_lens[0:lmax])/np.amax(cl_ee_lens_map)
    rescaleEENoise=l_vec[526]**2*n_l_ee[526]/n_l_ee_map[1515]
    print 'rescaleEENoise', rescaleEENoise
    rescaleEENoise=1
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_ee[0:lmax], label=r'$l(l+1)c^{EE}_l$ camb')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_ee_lens[0:lmax], label=r'$l(l+1)\tilde{c}^{EE}_l$ camb')
    
    plt.plot(l_arr, cl_ee_map*rescaleEE, label=r'$l(l+1)c^{EE}_l$ map')
    plt.plot(l_arr, cl_ee_lens_map*rescaleEElensed, label=r'$l(l+1)\tilde{c}^{EE}_l$ map')
    
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*n_l_ee[0:lmax], label=r'$l(l+1)n^{EE}_l$ theory')
    plt.plot(l_arr, n_l_ee_map*rescaleEENoise, label=r'$l(l+1)n^{EE}_l$ map')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=3)
    plt.yscale('log')
    plt.ylim(1e-9, 1e5)
    plt.title('E Mode Polarisation Power Spectra')
    #plt.savefig(saving_dir+'/Spectra/EE_spectra_beam'+str(round(sigma_am, 2))+'am_noise'+str(round(delta_t_am*sqrt(2), 2))+'.png')
    plt.xscale('log')
    plt.savefig(saving_dir+'/Spectra/EE_spectra_'+exp+'_'+str(nside)+'_loglog.png')
    #plt.savefig(saving_dir+'/Spectra/EE_spectra_'+exp+'.png')
    plt.show()
    plt.close()
    """

if spec=='eb':
    make_spec_plots('EB', E, B, E_lensed, B_lensed, l_vec, np.zeros(l_vec.shape), cl_ee_lens)#is cl ee lens=cl eb lens??, just proportional? #np.zeros(l_vec.shape))
    
    """"
    l_arr,cl_eb_map=cross_spectrum(E,B,fov_deg)
    l_arr,cl_eb_lens_map=cross_spectrum(E_lensed,B_lensed,fov_deg)
    plt.plot(l_arr, cl_eb_map, label=r'$l(l+1)c^{EB}_l$')
    plt.plot(l_arr, cl_eb_lens_map, label=r'$l(l+1)\tilde{c}^{EB}_l$')
    plt.legend(loc=4)
    plt.yscale('log')
    plt.title('EB Power Spectra from map')
    #plt.savefig(saving_dir+'/Spectra/EB_spectra_map_beam'+str(round(sigma_am, 2))+'am_noise'+str(round(delta_t_am*sqrt(2), 2))+'.png')
    plt.xscale('log')
    plt.savefig(saving_dir+'/Spectra/EB_spectra_'+exp+'_'+str(nside)+'_loglog.png')
    #plt.savefig(saving_dir+'/Spectra/EB_spectra_'+exp+'.png')
    plt.show()
    plt.close()
    """

if spec=='tb':
    make_spec_plots('TB', T, B, T_lensed, B_lensed, l_vec, np.zeros(l_vec.shape), cl_te_lens)#np.zeros(l_vec.shape))
    
    """"
    l_arr,cl_tb_map=cross_spectrum(T,B,fov_deg)
    l_arr,cl_tb_lens_map=cross_spectrum(T_lensed,B_lensed,fov_deg)
    plt.plot(l_arr, l_arr*(l_arr+np.ones(l_arr.shape))*cl_tb_map, label=r'$l(l+1)c^{TB}_l$')
    plt.plot(l_arr, l_arr*(l_arr+np.ones(l_arr.shape))*cl_tb_lens_map, label=r'$l(l+1)\tilde{c}^{TB}_l$')
    plt.legend(loc=4)
    plt.yscale('log')
    plt.title('TB Power Spectra from map')
    #plt.savefig(saving_dir+'/Spectra/TB_spectra_map_beam'+str(round(sigma_am, 2))+'am_noise'+str(round(delta_t_am*sqrt(2), 2))+'.png')
    plt.xscale('log')
    plt.savefig(saving_dir+'/Spectra/TB_spectra_'+exp+'_'+str(nside)+'_loglog.png')
    #plt.savefig(saving_dir+'/Spectra/TB_spectra_'+exp+'.png')
    plt.show()
    plt.close()
    """
    
if spec=='te':
    make_spec_plots('TE', T, E, T_lensed, E_lensed, l_vec, cl_te, cl_te_lens)
    
    """"
    l_arr,cl_te_map=cross_spectrum(T,E,fov_deg)
    l_arr,cl_te_lens_map=cross_spectrum(T_lensed,E_lensed,fov_deg)
    rescaleTE=np.amax(l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*np.absolute(cl_te[0:lmax]))/np.amax(np.absolute(cl_te_map))
    rescaleTElensed=np.amax(l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*np.absolute(cl_te_lens[0:lmax]))/np.amax(np.absolute(cl_te_lens_map))
    print 'factors by which unlensed and lensed TE spectra are off:', rescaleTE, rescaleTElensed
    rescaleTE=1
    rescaleTElensed=1
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*(cl_te[0:lmax]), label=r'$l(l+1)c^{TE}_l$ camb')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*(cl_te_lens[0:lmax]), label=r'$l(l+1)\tilde{c}^{TE}_l$ camb')

    plt.plot(l_arr, (cl_te_map*rescaleTE), label=r'$l(l+1)c^{TE}_l$ map')
    plt.plot(l_arr, (cl_te_lens_map*rescaleTElensed), label=r'$l(l+1)\tilde{c}^{TE}_l$ map')

    #plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*n_l_ee[0:lmax], label=r'$l(l+1)n^{EE}_l$')
    #plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*n_l_tt[0:lmax], label=r'$l(l+1)n^{TT}_l$')
    #plt.plot(l_arr, l_arr*(l_arr+np.ones(l_arr.shape))*n_l_ee_map, label=r'$l(l+1)n^{EE}_l$')
    #plt.plot(l_arr, l_arr*(l_arr+np.ones(l_arr.shape))*n_l_tt_map, label=r'$l(l+1)n^{TT}_l$')

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=2)
    #plt.yscale('log')
    plt.title('Temperature and E Mode Polarisation Cross Power Spectra')
    #plt.savefig(saving_dir+'/Spectra/TE_spectra_map_beam'+str(round(sigma_am, 2))+'am_noiseT'+str(delta_t_am)+'.png')
    plt.savefig(saving_dir+'/Spectra/TE_spectra_'+exp+'_'+str(nside)+'.png')
    plt.show()
    plt.close()
    """
    
if spec=='bb':
    make_spec_plots('BB', B, B, B_lensed, B_lensed, l_vec, np.zeros(l_vec.shape), cl_bb_lens)
    """"
    l_arr,cl_bb_map=power_spectrum(B,fov_deg)
    l_arr,cl_bb_lens_map=power_spectrum(B_lensed,fov_deg)
    print 'about to do bb plot'
    rescaleBBlensed=np.amax(l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_bb_lens[0:lmax])/np.amax(cl_bb_lens_map)
    print 'b camb and map spec ratio', rescaleBBlensed
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_bb_lens[0:lmax], label=r'$l(l+1)\tilde{c}^{BB}_l$ camb')
    plt.plot(l_arr,cl_bb_map, label=r'$l(l+1)c^{BB}_l$ map')
    plt.plot(l_arr,cl_bb_lens_map, label=r'$l(l+1)\tilde{c}^{BB}_l$ map')
    plt.plot(l_vec[0:lmax], l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*n_l_bb[0:lmax], label=r'$l(l+1)n^{BB}_l$ theory')
    plt.plot(l_arr, n_l_bb_map, label=r'$l(l+1)n^{BB}_l$ map')
    plt.legend(loc='lower left')
    plt.yscale('log')
    plt.title('B Mode Polarisation Power Spectra')
    #plt.savefig(saving_dir+'/Spectra/BB_spectra_map_beam'+str(round(sigma_am, 2))+'am_noise'+str(round(delta_t_am*sqrt(2), 2))+'.png')
    plt.xscale('log')
    plt.savefig(saving_dir+'/Spectra/BB_spectra_'+exp+'_'+str(nside)+'loglog.png')
    plt.show()
    plt.close()
    print 'bb plot done'
    """



#lensing field spectra
if est=='conv':
    conv_map=np.loadtxt(working_dir+'/LensingField/conv_in_'+spec+'_'+str(fov_deg)+'_'+str(nside))
elif est=='shear_plus':
    conv_map=np.loadtxt(working_dir+'/LensingField/shear_plus_in_'+spec+'_'+str(fov_deg)+'_'+str(nside))
elif est=='shear_cross':
    conv_map=np.loadtxt(working_dir+'/LensingField/shear_cross_in_'+spec+'_'+str(fov_deg)+'_'+str(nside))

l_arr, conv_spec=power_spectrum(conv_map,fov_deg)



conv_map_out=np.ones((nside,nside,len(num_maps)))
conv_map_from_noise=np.ones((nside,nside,len(num_maps)))
conv_map_from_unlensed=np.ones((nside,nside,len(num_maps)))

conv_out_spec=np.ones((l_arr.shape[0], len(num_maps)))
conv_out_spec_rescaled=np.ones((l_arr.shape[0], len(num_maps)))
conv_from_noise_spec=np.ones((l_arr.shape[0], len(num_maps)))
conv_from_unlensed_spec=np.ones((l_arr.shape[0], len(num_maps)))

l_arr_fac=l_arr*(l_arr+np.ones(l_arr.shape))/(2*pi)


fsky=(fov_deg*(pi/180.))**2/(4.*pi)
rescaleConvOut=((nside/4)**2/fsky)**4
for i, n in enumerate(num_maps):
    print i, n
    if est=='conv':
        conv_map_out[:,:,i]=np.loadtxt(working_dir+'/LensingField/conv_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(n)+'maps')
        #conv_map_from_noise[:,:,i]=np.loadtxt(working_dir+'/LensingField/conv_from_noise_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(n)+'maps')
        #conv_map_from_unlensed[:,:,i]=np.loadtxt(working_dir+'/LensingField/conv_from_unlensed_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(n)+'maps')
    elif est=='shear_plus':
        conv_map_out[:,:,i]=np.loadtxt(working_dir+'/LensingField/shear_plus_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(n)+'maps')
        #conv_map_from_noise[:,:,i]=np.loadtxt(working_dir+'/LensingField/shear_plus_from_noise_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(n)+'maps')
        #conv_map_from_unlensed[:,:,i]=np.loadtxt(working_dir+'/LensingField/shear_plus_from_unlensed_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(n)+'maps')
    elif est=='shear_cross':
        conv_map_out[:,:,i]=np.loadtxt(working_dir+'/LensingField/shear_cross_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(n)+'maps')
        #conv_map_from_noise[:,:,i]=np.loadtxt(working_dir+'/LensingField/shear_cross_from_noise_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(n)+'maps')
        #conv_map_from_unlensed[:,:,i]=np.loadtxt(working_dir+'/LensingField/shear_cross_from_unlensed_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(n)+'maps')
    plt.imshow(conv_map_out[:,:,i])
    plt.colorbar()
    plt.show()
    
    l_arr, conv_out_spec[:,i]=power_spectrum(conv_map_out[:,:,i],fov_deg)

    conv_map_out_rescaled=conv_map_out
    l_arr, conv_out_spec_rescaled[:,i]=power_spectrum(conv_map_out_rescaled[:,:,i],fov_deg)
    #conv_out_spec_rescaled*=4*pi    #kks paper
    conv_out_spec_rescaled[:,i]*=2*pi

    np.savetxt(datafile_saving_dir+'/output spec '+str(nside), conv_out_spec)
    np.savetxt(datafile_saving_dir+'/output spec rescaled '+str(nside), conv_out_spec_rescaled)
    np.savetxt(datafile_saving_dir+'/output spec ls '+str(nside), l_arr)
    
    l_arr, conv_from_noise_spec[:,i]=power_spectrum(conv_map_from_noise[:,:,i],fov_deg)
    
    l_arr, conv_from_unlensed_spec[:,i]=power_spectrum(conv_map_from_unlensed[:,:,i],fov_deg)

    #plt.plot(l_arr, np.sqrt((conv_out_spec[:,i]*rescaleConvOut).astype('float')), label=est+ ' out from '+str(n)+ ' '+spec+' map(s)')

    plt.plot(l_arr, np.sqrt(l_arr_fac*(conv_out_spec_rescaled[:,i]).astype('float')), label=est+ ' out from '+str(n)+ ' '+spec+' map(s)')

    #plt.plot(l_arr, conv_from_noise_spec[:,i]*rescaleConvOut, label=est+ ' out from '+str(n)+ ' '+spec+' noise map(s)')
    
    #plt.plot(l_arr, np.sqrt(conv_from_unlensed_spec[:,i]*rescaleConvOut), label=est+ ' out from '+str(n)+' unlensed '+spec+' map(s)')
    #plt.plot(l_arr, np.sqrt((conv_from_unlensed_spec[:,i]-conv_out_spec[:,i])*rescaleConvOut), label='from unlensed-total')
    #plt.plot(l_arr, np.sqrt(np.abs(conv_out_spec[:,i]-conv_from_unlensed_spec[:,i])*rescaleConvOut), label='output-unlensed')

Ls_N0=np.loadtxt(N0_directory+'/planck_N_'+(spec))  #upper(spec)
Ls=(Ls_N0[0, :]).astype(int)
N0dd=Ls_N0[1, :]
N0kk=Ls**4*N0dd
N0kk_spline=spline1d(Ls, N0kk)
if(subtract_N0):
    #read in and plot N0 corrections to lensing spectra
    
    N0kk_to_plot=Ls*(Ls+np.ones(Ls.shape))*N0kk/(2*pi)

    
    conv_in_spline=(l_vec, cl_kk)
    conv_in_Ls=conv_in_spline(Ls)

    l_arr_new, ind= np.unique(l_arr, return_index=True)
    ind[1]=2
    conv_out_spec_new=conv_out_spec[:,0].reshape(l_arr.shape)[ind]
    conv_out_spline=spline1d(l_arr_new, conv_out_spec_new)
    conv_out_Ls=conv_out_spline(Ls)



    conv_in_plus_noise=Ls*(Ls+np.ones(Ls.shape))*conv_in_Ls/(2*pi)+N0kk_to_plot
    print 'N0kk to print',N0kk_to_plot
    #change back
    conv_out_minus_noise=conv_out_Ls*rescaleConvOut-N0kk_to_plot
    #conv_minus_noise=conv_in_Ls_to_plot-N0kk_to_plot
    print conv_out_minus_noise

    plt.plot(Ls, np.sqrt(conv_in_plus_noise), label='convergence in from CAMB + N0  ')
    plt.plot(Ls, np.sqrt(np.abs(conv_out_minus_noise)), label='convergence out from map - N0')
    plt.plot(Ls, np.sqrt(N0kk_to_plot), label='N0')
    #plt.plot(Ls, np.sqrt(conv_minus_noise), label='convergence in from CAMB - N0')
    #rescaleConvOut=Ls*(Ls+np.ones(Ls.shape))*conv_in_plus_noise/conv_out_Ls.reshape((19,))

if est=='conv':
    plt.plot(l_vec[0:lmax], np.sqrt(l_vec_fac*cl_kk[0:lmax]), label='convergence in from CAMB')
else:
    plt.plot(l_vec[0:lmax], np.sqrt(l_vec_fac/2*cl_kk[0:lmax]), label='shear in from CAMB')


plt.plot(l_arr, np.sqrt(l_arr_fac*conv_spec), label=est+ ' in from map')


#rescaleConvOut=(l_vec[0:lmax]*(l_vec[0:lmax]+np.ones(l_vec[0:lmax].shape))*cl_kk[0:lmax])[526]/(conv_out_spec[1515])

print 'rescale factor reqd for output convergence:', rescaleConvOut
#plt.plot(l_arr, conv_spec*rescaleConv*rescaleConv2, label=est+ ' in from '+spec+' map renormalised')




#plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=2)
plt.legend(loc='upper center', ncol=2)
plt.xlim(10,1000)
plt.ylim(0, 0.06)#plt.ylim(1e-3, 1e-1)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('L')
plt.ylabel(r'$[l(l+1)C^{\kappa\kappa}_L/2\pi]^{1/2}$')
plt.title('Lensing Power Spectra - Real Space')
#plt.savefig(saving_dir+'/Spectra/'+est+'_'+spec+'_spectra_beam'+str(round(sigma_am, 2))+'am_noise'+str(delta_t_am)+'.png')
plt.savefig(saving_dir+'/Spectra/'+est+'_'+spec+'_spectra_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+'.png')
plt.show()
plt.close()

#form fac from spectra
conv_20=np.loadtxt(working_dir+'/LensingField/conv_out_with_bias'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps[1])+'maps')
l, conv_20_spec=power_spectrum(conv_20, fov_deg)

plt.plot(l_arr, np.sqrt(l_arr_fac*conv_20_spec.astype('float')), label='from '+str(n)+' maps with bias')
plt.plot(l_arr, np.sqrt(l_arr_fac*(conv_out_spec_rescaled[:,1]).astype('float')), label='from '+str(n)+' maps with bias removed')
plt.legend()
plt.xlim(10,1000)
#plt.ylim(1e-8,1e-5)
plt.xlabel('L')
plt.ylabel(r'$[l(l+1)C^{\kappa\kappa}_L/2\pi]^{1/2}$')
plt.xscale('log')
plt.yscale('log')
plt.show()

ff=conv_out_spec_rescaled[:,1].astype('float')/conv_spec
plt.plot(l_arr, ff)
plt.title('Form Factor from simulated spectra')
plt.xlim(0,1000)
plt.ylim(0,1.5)
plt.savefig(saving_dir+'/FormFactor/'+spec+'_'+est+'_'+exp+'_form_factor_from_spectra')
plt.show()
plt.close()

Lff=np.zeros((l_arr.shape[0],2))
Lff[:,0]=l_arr
Lff[:,1]=ff
print Lff
np.savetxt(saving_dir+'/FormFactor/'+spec+'_'+est+'_'+exp+'_form_factor_from_spectra_data_file', Lff)


#bias from lensed map with randomised phases
binwidth=10
l_bias=np.arange(2, l_arr[l_arr.size-1], binwidth)
bias_power=np.empty(l_bias.shape, dtype=complex)
for j in range(num_rand_phase_maps):
    bias=np.loadtxt(working_dir+'/LensingField/'+est+'_bias_from_rand_phases_'+str(j)+'_'+spec+'_'+str(fov_deg))
    l_arr_bias, ps_bias=power_spectrum(bias, fov_deg, binsize=binwidth)
    bias_power+=ps_bias
bias_power/=num_rand_phase_maps

#bias power from unlensed maps as a cross check
bias_power_unlensed=np.empty(l_bias.shape, dtype=complex)
for i in range(num_rand_phase_maps):
    bias=np.loadtxt(working_dir+'/LensingField/'+est+'_bias_from_unlensed_map_'+str(i)+'_'+spec+'_'+str(fov_deg))
    l_arr_bias, ps_bias=power_spectrum(bias, fov_deg, binsize=binwidth)
    bias_power_unlensed+=ps_bias
bias_power_unlensed/=num_rand_phase_maps





kappa1=np.loadtxt(working_dir+'/LensingField/'+est+'_from_1_map_'+spec+'_'+str(fov_deg))
l_arr, kappa1_spec=power_spectrum(kappa1, fov_deg, binsize=10)


l_arr_bias, kappa1_spec_bias=power_spectrum(kappa1, fov_deg, binsize=binwidth)
conv_spec_rand_phase=kappa1_spec_bias-bias_power
conv_spec_unlensed_bias=kappa1_spec_bias-bias_power_unlensed
print conv_spec_rand_phase
print conv_spec_unlensed_bias

kappa1_spec*=2*pi
kappa1_spec_bias*=2*pi
conv_spec_rand_phase*=2*pi
conv_spec_unlensed_bias*=2*pi

bias_power_unlensed*=2*pi
bias_power*=2*pi

plt.plot(l_vec[0:lmax], cl_kk[0:lmax], label='convergence in from CAMB')
plt.plot(l_arr_bias[1:], bias_power[1:].astype('float'), label='random phase bias power')#, marker='o', linestyle='None')
plt.plot(l_arr_bias[1:], bias_power_unlensed[1:].astype('float'), label='unlensed bias power')
plt.plot(l_arr, (kappa1_spec.astype('float')), label='from 1 map with bias')
plt.plot(l_arr, (conv_spec), label=est+ ' in from map')
plt.legend(loc='lower center', ncol=2)
plt.xlim(10,1000)
plt.ylim(1e-8,1e-5)
plt.xlabel('L')
plt.ylabel(r'$[l(l+1)C^{\kappa\kappa}_L/2\pi]^{1/2}$')
plt.xscale('log')
plt.yscale('log')
plt.title('Lensing Power Spectra - Real Space')
#plt.savefig(saving_dir+'/Spectra/'+est+'_'+spec+'_spectra_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+'with'+str(num_rand_phase_maps)+'randphasemaps.png')
plt.show()
plt.close()

N0_bias=N0kk_spline(l_arr_bias)
error=np.sqrt(N0_bias/(2*pi*np.sqrt(2*binwidth+1)))


plt.plot(l_vec, cl_kk, label='c(l)')
plt.plot(l_arr_bias, N0_bias, label='N(l)')
plt.plot(l_arr_bias, error, label='error(l)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, 10000)
plt.legend(loc='lower center')
plt.show()
plt.close()


plt.plot(l_vec[0:lmax], np.sqrt(l_vec_fac*cl_kk[0:lmax]), label='convergence in from CAMB')
#plt.plot(l_arr, np.sqrt(l_arr_fac*conv_spec), label=est+ ' in from map')
plt.plot(l_arr, np.sqrt(l_arr_fac*(kappa1_spec.astype('float'))), label='from 1 map with bias')
#plt.errorbar(l_arr_bias, np.sqrt(np.abs(l_arr_bias*(l_arr_bias+1)/(2*pi)*(conv_spec_rand_phase.astype('float')))), label=est+ ' out minus rand phase bias', yerr=error, fmt='-o')
plt.plot(l_arr_bias, conv_spec_rand_phase.astype('float'), label='random phase bias removed')#, marker='o', linestyle='None')
plt.plot(l_arr_bias, conv_spec_rand_phase.astype('float'), label='unlensed bias power removed')
#plt.plot(l_arr_bias, np.sqrt(np.abs(l_arr_bias*(l_arr_bias+1)/(2*pi)*(conv_spec_rand_phase.astype('float')))), label=est+ ' out rand phase bias '+str(num_rand_phase_maps)+' maps')
#plt.plot(l_bin, np.sqrt(np.abs(l_bin*(l_bin+1)/(2*pi)*(conv_spec_rand_phase_binned.astype('float')))), label=est+ ' out with random phase bias '+str(num_rand_phase_maps)+' maps binned')
#plt.plot(l_arr, np.sqrt(l_arr_fac*(conv_out_spec_rescaled[:,0]).astype('float')), label=est+ ' out minus actual bias')
for i, n in enumerate(num_maps):
    if n==1:
        plt.plot(l_arr, np.sqrt(l_arr_fac*(conv_out_spec_rescaled[:,i]).astype('float')), label='from 1 map with bias removed')
    else:
        plt.plot(l_arr, np.sqrt(l_arr_fac*(conv_out_spec_rescaled[:,i]).astype('float')), label='from '+str(n)+' maps with bias removed')

plt.legend(loc='lower center', ncol=2)
plt.xlim(10,1000)
plt.ylim(1e-3, 1e-1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('L')
plt.ylabel(r'$[l(l+1)C^{\kappa\kappa}_L/2\pi]^{1/2}$')
plt.title('Lensing Convergence from TT estimator')#('Lensing Power Spectra - Real Space')
plt.savefig(saving_dir+'/Spectra/'+est+'_'+spec+'_spectra_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+'with'+str(num_rand_phase_maps)+'randphasemaps.png')
plt.show()
plt.close()




#For harmonic space random phase bias:
"""if spec=='tt':
    bias_power_HS=np.empty(l_bias.shape, dtype=complex)
    for j in range(num_rand_phase_maps):
        if spec=='tt':
            bias=np.loadtxt(HS_dir+'/LensingField/conv_bias_from_rand_phases_'+str(j)+'_'+spec+'_'+str(fov_deg))
            l_arr_bias, ps_bias=power_spectrum(bias, fov_deg, binsize=binwidth)
            bias_power_HS+=ps_bias
    bias_power_HS/=num_rand_phase_maps

    kappa1_HS=np.loadtxt(HS_dir+'/LensingField/conv_from_1_map_HS_'+str(fov_deg)+'_'+str(nside))
    l_arr, kappa1_spec_HS=power_spectrum(kappa1_HS, fov_deg)

    kappa1_wo_bias=np.loadtxt(HS_dir+'/LensingField/conv_out_HS_'+str(fov_deg)+'_'+str(nside)+'_20maps')
    l_arr, kappa1_spec_wo_bias_HS=power_spectrum(kappa1_wo_bias, fov_deg)

    kappa20_HS=np.loadtxt(HS_dir+'/LensingField/conv_out_HS_'+str(fov_deg)+'_'+str(nside)+'_20maps')
    l_arr, kappa20_spec_HS=power_spectrum(kappa20_HS, fov_deg)



    conv_spec_rand_phase_HS=kappa1_spec_HS-bias_power_HS
    print conv_spec_rand_phase_HS


    #kappa1_spec_HS*=2*pi
    #kappa1_spec_wo_bias_HS*=2*pi
    #kappa20_spec_HS*=2*pi
    #conv_spec_rand_phase_HS*=2*pi



    plt.plot(l_arr, np.sqrt(l_arr_fac*(kappa1_spec_HS.astype('float'))), label=est+ ' out from 1 '+spec+' map')
    plt.plot(l_arr, np.sqrt(np.abs(l_arr_fac*(conv_spec_rand_phase_HS.astype('float')))), label=est+ ' out with random phase bias '+str(num_rand_phase_maps)+' maps')
    #plt.plot(l_bin, np.sqrt(np.abs(l_bin*(l_bin+1)/(2*pi)*(conv_spec_rand_phase_binned_HS.astype('float')))), label=est+ ' out with random phase bias '+str(num_rand_phase_maps)+' maps')
    #plt.plot(l_arr, np.sqrt(l_arr_fac*kappa20_spec_HS.astype('float')), label=est+ ' out from 20 maps')
    plt.plot(l_arr, np.sqrt(l_arr_fac*kappa1_spec_wo_bias_HS.astype('float')), label=est+ ' out from 1 map actual bias')
    plt.plot(l_vec[0:lmax], np.sqrt(l_vec_fac*cl_kk[0:lmax]), label='convergence in from CAMB')
    plt.plot(l_arr, np.sqrt(l_arr_fac*conv_spec), label=est+ ' in from map')
    plt.legend(loc='lower center', ncol=2)
    plt.xlim(10,1000)
    plt.ylim(1e-3, 1e-1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('L')
    plt.ylabel(r'$[l(l+1)C^{\kappa\kappa}_L/2\pi]^{1/2}$')
    plt.title('Lensing Power Spectra - Harmonic Space')
    plt.savefig(saving_dir+'/Spectra/'+est+'_'+spec+'_spectra_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+'with'+str(num_rand_phase_maps)+'randphasemapsHS.png')
    plt.show()
    plt.close()
"""
#trying to plot a histogram of values of convergence map to see if scale is right (2 arcmin deflection angle)
"""
conv_map.shape=nside**2
plt.hist((conv_map), bins=31)
plt.xlabel('convergence')
plt.ylabel('number of pixels with a certain kappa value')
plt.title('input map')
plt.savefig(saving_dir+'/PixelValues/'+est+'_'+spec+'_histogram_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+' input.png')
plt.show()



plt.hist((conv_map_out), bins=41)
plt.xlabel('convergence')
plt.ylabel('number of pixels with a certain kappa value')
plt.title('output map')
plt.savefig(saving_dir+'/PixelValues/'+est+'_'+spec+'_histogram_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+' reconstructed.png')
plt.show()


conv_map_in_rescaled=conv_map#*nside
plt.hist((conv_map_in_rescaled), bins=31)
plt.xlabel('convergence')
plt.ylabel('number of pixels with a certain kappa value')
plt.title('rescaled input map')
plt.savefig(saving_dir+'/PixelValues/'+est+'_'+spec+'_histogram_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+' input.png')
plt.show()
"""

plt.imshow(conv_map, origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu')
plt.colorbar()
plt.title('Convergence Field')
plt.savefig(saving_dir+'/LensingFieldPics/convergence_'+str(nside)+'_'+str(int(fov_deg))+'.png')
plt.show()

conv_map.shape=nside**2

for i, n in enumerate(num_maps):
    conv_map_out_n=conv_map_out[:,:,i]
    conv_map_out_n.shape=nside**2
    conv_map_out_rescaled=conv_map_out_n#*nside**4/A

    bins=np.linspace(-5e-1, 5e-1, 100)  #(-1e-6, 1e-6, 100)
    plt.hist((conv_map_out_rescaled), bins, alpha=0.5,label='reconstructed')
    plt.hist((conv_map), bins, alpha=0.5, label='input')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend(loc='upper right')
    plt.xlabel('convergence')
    plt.ylabel('number of pixels with a certain kappa value')
    plt.title('Convergence Maps Pixel Histogram for '+str(n)+' Maps')
    plt.savefig(saving_dir+'/PixelValues/'+est+'_'+spec+'_histogram_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+'_'+str(n)+'maps.png')
    plt.show()

"""
plt.plot(l_vec[0:lmax], cl_kk[0:lmax], label='conv from camb')
plt.plot(l_vec[0:lmax], cl_dd[0:lmax]*l_vec[0:lmax]*(l_vec[0:lmax]+1)/4, label='conv from defl from camb')
plt.legend(loc='lower center', ncol=2)
plt.xscale('log')
plt.title('convergence spectrum')
plt.show()





deflection_map=np.loadtxt(working_dir+'/LensingField/deflection_map_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))




fov_rad=(pi/180.)*fov_deg
delta_theta=fov_rad/nside
deflection_map*=delta_theta

deflection_map_am=deflection_map*180/pi*60

plt.title('Deflection Field')
plt.imshow(deflection_map_am, origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='YlOrRd')
plt.colorbar()
plt.savefig(saving_dir+'/LensingFieldPics/deflection_angle_magnitude_'+str(nside)+'_'+str(int(fov_deg))+'.png')
plt.show()

l_arr, deflection_spec=power_spectrum(deflection_map,fov_deg)

plt.plot(l_arr, np.sqrt(l_arr_fac*deflection_spec), label='from map')
plt.plot(l_arr, np.sqrt(l_arr_fac/(l_arr*(l_arr+1))*4*conv_spec), label='from conv spec from map')
plt.plot(l_vec[0:lmax], np.sqrt(l_vec_fac*cl_dd[0:lmax]), label='from camb')
plt.xlim(1,10000)
#plt.ylim(1e-6, 1e-3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('L')
plt.ylabel(r'$[l(l+1)C^{\alpha\alpha}_L/2\pi]^{1/2}$')
plt.legend(loc='lower center', ncol=2)
plt.title('Deflection Angle Power Spectrum')
plt.savefig(saving_dir+'/Spectra/deflection_angle_spectrum_'+exp+'_'+str(nside)+'_'+str(int(fov_deg))+'.png')
plt.show()
plt.close()

deflection_map.shape=nside**2
bins=np.linspace(0, 3e-3, 100)
#bins=np.linspace(0, 8e-7, 100)
plt.hist(deflection_map, bins)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(loc='upper right')
plt.xlabel('deflection angle')
plt.ylabel('number of pixels')
plt.title('Deflection Maps Pixel Histogram for '+str(n)+' Maps')
plt.savefig(saving_dir+'/PixelValues/deflection_angle_histogram_'+str(nside)+'_'+str(int(fov_deg))+'.png')
plt.show()

plt.imshow(T)
plt.title('Temperature')
plt.colorbar()
plt.show()

T.shape=nside**2
bins=np.linspace(-250, 250, 100)
#bins=np.linspace(0, 8e-7, 100)
plt.hist(T, bins)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(loc='upper right')
plt.xlabel('deflection angle')
plt.ylabel('number of pixels')
plt.title('Temperature Maps Pixel Histogram for '+str(n)+' Maps')
plt.savefig(saving_dir+'/PixelValues/deflection_angle_histogram_'+str(nside)+'_'+str(int(fov_deg))+'.png')
plt.show()
"""

