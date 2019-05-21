#run this to perform the reconstruction
#This code takes nside (number of pixels per side in map) and fov_deg (field of view in degrees) as command-line arguments
#NB! Change working_dir variable in this code and make sure the working dierectory has a folder called Reconstructions and one called LensingField
# Run as follows in ipython: import sys ; sys.argv=['reconstruct_shear_many_maps_with_noise.py','2048','20'] ; execfile("reconstruct_shear_many_maps_with_noise.py")
# For MATHEW MAPS: Run as follows in ipython: import sys ; sys.argv=['reconstruct_shear_many_maps_with_noise.py','1200','20'] ; execfile("reconstruct_shear_many_maps_with_noise.py")
# For MATHEW MAPS: Run as follows in ipython: import sys ; sys.argv=['reconstruct_shear_many_maps_with_noise.py','600','10'] ; execfile("reconstruct_shear_many_maps_with_noise.py")

import numpy as np
from math import *
import pylab
#from IPython.Shell import IPShellEmbed
import map_making_functions
import matplotlib.pyplot as plt
import estimators
import sys

from astropy.io import fits

import inputs
import os

# USING nside = 2048 and 20 deg

# NO OPTION TO USE FOREGROUND BASED FILTERS ?

use_Mathew_maps = False#True

spec='tt'#'tt'  #should be tt, ee, tb, eb, te - make sure you have made the filter for this spectrum by running make_real_space_filter.py with this value for spec
exp=exp_filt='advact' # 'advact'  'advact'   # planck, ref, advact
est='conv'  # conv, shear_plus, shear_cross, add shear_E and shear_B???
use_noisy_maps=False#True #If True, the E and B maps obtained from noisy Q and U maps will be used, if this is False, then the reconstructions will be applied to noise-free maps which is obviously unrealistic but useful for testing.
use_noisy_unlensed=False#True
num_maps=1
num_rand_phase_maps=0

args=['reconstruct_shear_many_maps_with_noise.backup.18April.py','1024','20'] ;
nside=int(args[1])
fov_deg=float(args[2])


#to get zero to be white in plots
from matplotlib.colors import Normalize
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
norm = MidpointNormalize(midpoint=0)


if est=='conv':
    est_name='Convergence'
elif est=='shear_plus':
    est_name='Shear Plus'
elif est=='shear_cross':
    est_name='Shear Cross'
elif est=='shear_E':
    est_name='Shear E'
elif est=='shear_B':
    est_name='Shear B'

RS_cut_size=nside/8


if exp_filt=='planck':
    sigma_am=7.
    delta_t_am=27.
    delta_p_am=40*sqrt(2)

if exp=='act':
    delta_t_am=15
    delta_p_am=delta_t_am*sqrt(2)
    sigma_am=1.5

if exp_filt=='ref':
    sigma_am=1.5 # 1.4#60*fov_deg/nside
    delta_t_am= 1.5 # 2. 
    delta_p_am=delta_t_am*sqrt(2)

# KM UPDATED REF TO RUN ON SIMULATED ACTPOL MAPS

elif (exp[0:3]=='ref'):
    delta_t_am=float(exp[3:])
    delta_p_am=delta_t_am*sqrt(2)
    sigma_am=1.

if exp=='advact':
    delta_t_am=7
    delta_p_am=delta_t_am*sqrt(2)
    sigma_am=1.4


noise_free_map_dir= inputs.working_dir_base + exp +'/fov'+str(int(fov_deg)) # '/Users/Heather/Documents/HeatherVersionPlots/'+exp+'/fov'+str(int(fov_deg))

if not os.path.exists(noise_free_map_dir):
	os.makedirs(noise_free_map_dir)	

if (exp[0:3]=='ref'):
    working_dir=inputs.working_dir_base + 'ref' + '/fov'+ str(int(fov_deg))
    if not os.path.exists(working_dir):
		os.makedirs(working_dir)
else:
    working_dir=inputs.working_dir_base + exp +'/fov'+str(int(fov_deg))     #+'/jetnormconv'
    if not os.path.exists(working_dir):
		os.makedirs(working_dir)

working_dir_Reconstructions=inputs.working_dir_base+exp+'/fov'+str(int(fov_deg))+'/Reconstructions'
if not os.path.exists(working_dir_Reconstructions):
	os.makedirs(working_dir_Reconstructions)

working_dir_LensingField=inputs.working_dir_base+exp+'/fov'+str(int(fov_deg))+'/LensingField'
if not os.path.exists(working_dir_LensingField):
	os.makedirs(working_dir_LensingField)


fudge = inputs.fudge # fudge=1.0 # in inputs.py now

print 'nside', nside
print 'fov', fov_deg

#This band-pass function defines which l-modes we plot for the convergence or shear map plots at the end.
def band_pass_fun(l):
    lmin= 30#>>15#15#60#15 * 15./fov_deg
    lmax= 200#>>100#150#100#200 * 15./fov_deg
    if l < lmin:
        return(0.)
    if l > lmax:
        return(0.)
    return(1.)

l_array=map_making_functions.make_l_array(nside, fov_deg)
lx,ly=map_making_functions.make_lxy_arrays(nside,fov_deg)

exp_print = exp + '_noise_'+str(round(delta_t_am,2))+'_'+'beam_'+str(round(sigma_am,2))
spec_print = spec + '_'+str(nside)+'_'+str(int(fov_deg))

# Load in the filter for the chosen estimator, sort out code to make plus and minus filters
if spec=='plus':
    spec_filt='ee'
elif spec=='minus' and est=='conv':
    spec_filt='ee'
elif spec=='cross':
    spec_filt='te' #Not actually used
else:
    spec_filt=spec
if est=='conv':
    ideal_filter=np.loadtxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_conv_filt')
    ideal_filter_RS=np.loadtxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_conv_filt_RS')
    #ideal_filter=np.loadtxt(working_dir+'/Filters/'+exp_filt+'_'+spec_filt+'_conv_filt_'+str(fov_deg)+'_'+str(nside))#+'_beam'+str(round(sigma_am, 2))+'_noise'+str(delta_t_am)) #Heather added _conv_
    #ideal_filter_RS=np.loadtxt(working_dir+'/Filters/'+exp_filt+'_'+spec_filt+'_conv_filt_RS_'+str(fov_deg)+'_'+str(nside))
elif est=='shear_plus':
    ideal_filter=np.loadtxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_plus_filt')
    ideal_filter_RS=np.loadtxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_plus_filt_RS')
    #ideal_filter=np.loadtxt(working_dir+'/Filters/'+exp_filt+'_'+spec_filt+'_shear_plus_filt_'+str(fov_deg)+'_'+str(nside))#+'_beam'+str(round(sigma_am, 2))+'_noise'+str(delta_t_am))
    #ideal_filter_RS=np.loadtxt(working_dir+'/Filters/'+exp_filt+'_'+spec_filt+'_shear_plus_filt_RS_'+str(fov_deg)+'_'+str(nside))
elif est=='shear_cross':
    ideal_filter=np.loadtxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_cross_filt')
    ideal_filter_RS=np.loadtxt(working_dir+'/Filters/'+exp_print+'_'+spec_print+'_shear_cross_filt_RS')
    #ideal_filter=np.loadtxt(working_dir+'/Filters/'+exp_filt+'_'+spec_filt+'_shear_cross_filt_'+str(fov_deg)+'_'+str(nside))#+'_beam'+str(round(sigma_am, 2))+'_noise'+str(delta_t_am))
    #ideal_filter_RS=np.loadtxt(working_dir+'/Filters/'+exp_filt+'_'+spec_filt+'_shear_cross_filt_RS_'+str(fov_deg)+'_'+str(nside))

map_print =  '_'+str(nside)+'_'+str(int(fov_deg))

# Load the lensing field (convergence, shear_plus or shear_cross)
print 'making lensing field'

# TAKE CARE OF PIXELISATION IN READING/WORKING WITH/CONVERTING MATHEW MAPS IN CAR PIXELISATION
if (use_Mathew_maps): 
#	hdul = fits.open('Plots_Data/ref/fov20/Lensing_Maps_Mathew_Hippo/' + 'run4_kappa_0000_pellmax5000_iau.fits')
	hdul = fits.open('Plots_Data/ref/fov20/Lensing_Maps_Mathew_Hippo/' + 'run1_kappa_0000.fits')
	conv_map_in=hdul[0].data
	conv_map = np.zeros(shape=(nside,nside))
	conv_map[:conv_map_in.shape[0],:conv_map_in.shape[1]] =  conv_map_in
	
else:
	conv_map=np.loadtxt(working_dir+'/Maps/conv_in'+map_print)

## WHEN RERUN MAKE_MAPS code ADD in shear_map_plus/cross to be made, then read in here instead of calling map_making_functions.conv_to_shear <<<<<
## >>> HAVE TO RERUN MAKE_MAPS to GET CORRECT shear_plus_in - mistakenly printed out conv_in instead of shear_plus_in  <<<<<<<<
# shear_map_plus = np.loadtxt(working_dir+'/Maps/shear_plus_in'+map_print) 
# shear_map_cross = np.loadtxt(working_dir+'/Maps/shear_cross_in'+map_print)

shear_map_plus, shear_map_cross=map_making_functions.conv_to_shear(conv_map,nside) 
## KAVI - CHECK THESE AGAINST MATHEW MAPS FOR REF EXPT

shear_E, shear_B=map_making_functions.shear_plus_cross_to_E_B(shear_map_plus, shear_map_cross, nside, fov_deg)

# Initialize the recovered map
kappa_sum=np.zeros(shape=(nside,nside))
kappa_sum_with_bias=np.zeros(shape=(nside,nside))
kappa_RS_sum=np.zeros(shape=(nside,nside))#np.zeros(shape=(nside-2*nside/8+1,nside-2*nside/8+1))
kappa_RS_sum_with_bias=np.zeros(shape=(nside,nside))
#kappa_noise_sum=np.zeros(shape=(nside,nside))

#####raise KeyboardInterrupt

# Run through input CMB maps - lensed and unlensed

for i in range(num_maps):
    print(i)
    """
    T=np.loadtxt(working_dir+'/Maps/T_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))
    E=np.loadtxt(working_dir+'/Maps/E_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))
    B=np.loadtxt(working_dir+'/Maps/B_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))
    """
    
    """
    plt.imshow(T)
    plt.colorbar()
    plt.show()
    plt.imshow(E)
    plt.colorbar()
    plt.show()
    plt.imshow(B)
    plt.colorbar()
    plt.show()
    """
 	    
    if(use_noisy_maps): 
		hdul = fits.open('Plots_Data/ref/fov20/Lensing_Maps_Mathew_Hippo/' + 'run2_obs_I_0000_pellmax5000_iau.fits')
		T_lensed_in=hdul[0].data
		T_lensed = np.zeros(shape=(nside,nside))		
		T_lensed[:T_lensed_in.shape[0],:T_lensed_in.shape[1]] =  T_lensed_in
        
        # COMMENT THIS IN IF USING OWN SIM MAPS NOT MATHEWS
		#	T_lensed=np.loadtxt(working_dir+'/Maps/T_lensed_with_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge))
		#	E_lensed=np.loadtxt(working_dir+'/Maps/E_lensed_with_QU_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge))
		#	B_lensed=np.loadtxt(working_dir+'/Maps/B_lensed_with_QU_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge))
	    #    Q_lensed=np.loadtxt(working_dir+'/Maps/Q_lensed_with_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge))
	    #    U_lensed=np.loadtxt(working_dir+'/Maps/U_lensed_with_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge))
        
		#plt.imshow(T_lensed); plt.colorbar(); plt.show()       
        

        #T_lensed=np.loadtxt(working_dir+'/Maps/T_lensed_with_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #E_lensed=np.loadtxt(working_dir+'/Maps/E_lensed_with_QU_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #B_lensed=np.loadtxt(working_dir+'/Maps/B_lensed_with_QU_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #Q_lensed=np.loadtxt(working_dir+'/Maps/Q_lensed_with_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #U_lensed=np.loadtxt(working_dir+'/Maps/U_lensed_with_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
		"""
        just_noise_T=np.loadtxt(working_dir+'/Maps/just_noise_T_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #next bit is just so it will run, not legit noise from Q and U maps combined
        just_noise_E=np.loadtxt(working_dir+'/Maps/just_noise_E_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        just_noise_B=np.loadtxt(working_dir+'/Maps/just_noise_B_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        """
    else:   #just use lensed map, no noise added
        T_lensed=np.loadtxt(noise_free_map_dir+'/Maps/T_lensed_'+str(i)+map_print+'_'+str(fudge))
        ##E_lensed=np.loadtxt(noise_free_map_dir+'/Maps/E_lensed_'+str(i)+map_print+'_'+str(fudge))
        ##B_lensed=np.loadtxt(noise_free_map_dir+'/Maps/B_lensed_'+str(i)+map_print+'_'+str(fudge))        
        ##U_lensed= np.loadtxt(working_dir+'/Maps/U_lensed_'+str(i)+map_print+'_'+str(fudge))
        ##Q_lensed= np.loadtxt(working_dir+'/Maps/Q_lensed_'+str(i)+map_print+'_'+str(fudge))

        #T_lensed=np.loadtxt(noise_free_map_dir+'/Maps/T_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #E_lensed=np.loadtxt(noise_free_map_dir+'/Maps/E_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #B_lensed=np.loadtxt(noise_free_map_dir+'/Maps/B_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #U_lensed= np.loadtxt(working_dir+'/Maps/U_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #Q_lensed= np.loadtxt(working_dir+'/Maps/Q_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))

        """
        just_noise_T=np.zeros((nside, nside))
        just_noise_E=np.zeros((nside, nside))
        just_noise_B=np.zeros((nside, nside))
        """
    if use_noisy_unlensed:
        T=np.loadtxt(working_dir+'/Maps/T_with_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge))
        ##E=np.loadtxt(working_dir+'/Maps/E_with_QU_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge))
        ##B=np.loadtxt(working_dir+'/Maps/B_with_QU_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge))
        # If needed Q and U with noise are available to be printed out in make_maps_with_noise.py line 305 and to be read in above

        #T=np.loadtxt(working_dir+'/Maps/T_with_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #E=np.loadtxt(working_dir+'/Maps/E_with_QU_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        #B=np.loadtxt(working_dir+'/Maps/B_with_QU_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
    else:
        # KAVI ADDED AS QUICK FIX for Mathew's inputs
        T = T_lensed 
        #T=np.loadtxt(noise_free_map_dir+'/Maps/T_'+str(i)+map_print)
        #E=np.loadtxt(noise_free_map_dir+'/Maps/E_'+str(i)+map_print)
        #B=np.loadtxt(noise_free_map_dir+'/Maps/B_'+str(i)+map_print)
        #Q=np.loadtxt(working_dir+'/Maps/Q_'+str(i)+map_print)
        #U=np.loadtxt(working_dir+'/Maps/U_'+str(i)+map_print)

        #T=np.loadtxt(noise_free_map_dir+'/Maps/T_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))
        #E=np.loadtxt(noise_free_map_dir+'/Maps/E_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))
        #B=np.loadtxt(noise_free_map_dir+'/Maps/B_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))
        
        #Q=np.loadtxt(working_dir+'/Maps/Q_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))
        #U=np.loadtxt(working_dir+'/Maps/U_'+str(i)+'_'+str(fov_deg)+'_'+str(nside))

# Do Reconstruction  # Kavi
# NOTES:
# Take into account impact of filter mismatch when computing cross-spectrum between input and recovered (also decide which kappa fid spectrum to us)
# Take care of pixelisation effects
# Direct full mode plot comparison ; band pass plot comparison
# Cross-spectrum
# Improve filter (more compact) and convolution edge effects for RS recovery

    import datetime
    from datetime import datetime
    print 'Start time', str(datetime.now())

    if spec=='tt':
        kappa=estimators.real_space_estimator(T_lensed, ideal_filter)
        kappa_sum_with_bias+=kappa
        bias=estimators.real_space_estimator(T, ideal_filter)
        kappa-=bias
        kappa_sum+=kappa
        np.savetxt(working_dir+'/LensingField/'+est+'_HS_kappa_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_iteration'+str(i),np.real(kappa))
        np.savetxt(working_dir+'/LensingField/'+est+'_HS_bias_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_iteration'+str(i)+'_noisefreeunlensedmap',np.real(bias))
        np.savetxt(working_dir+'/LensingField/'+est+'_HS_kappa_plus_bias_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_iteration'+str(i)+'_noisefreeunlensedmap',np.real(kappa_sum_with_bias))
		
        print 'Finished HS (FFT) based reconstruction', str(datetime.now())

        hdul = fits.open('Plots_Data/ref/fov20/Lensing_Maps_Mathew_Hippo/' + 'run2_recon_TT_0000_pellmax5000_iau.fits')
        kappa_Q_rec_in=hdul[0].data
        kappa_Q_rec = np.zeros(shape=(nside,nside))
        kappa_Q_rec[:kappa_Q_rec_in.shape[0],:kappa_Q_rec_in.shape[1]] =  kappa_Q_rec_in			


        #np.savetxt(working_dir+'/LensingField/'+est+'_bias_from_unlensed_map_'+str(i)+'_'+spec+'_'+str(fov_deg),np.real(bias))

		#### @@@ PLAN: Comment out MS reconstruction - run through HS reconstruction with updated filenames and all outputs - then come back to do MS reconstr #### 

        kappa_RS=estimators.real_space_RS_estimator(T_lensed, ideal_filter_RS)
        kappa_RS_sum_with_bias+=kappa_RS
        bias_RS=estimators.real_space_RS_estimator(T, ideal_filter_RS)
        kappa_RS-=bias_RS   #This is not realistic because in reality we don't know the exact unlensed map. We need to remove the bias differently for realistic reconstructions
        kappa_RS_sum+=kappa_RS
        print 'Finished MS (convolution) based reconstruction', str(datetime.now()) # Looks like 1.5 hours for 2 convolutions 2048^2 x 2048^2-> Speed up !!


        band_pass_filter=map_making_functions.make_filter(band_pass_fun, fov_deg, nside)
        conv_map_smoothed=map_making_functions.apply_filter(conv_map,band_pass_filter)
        kappa_Q_rec_smoothed=map_making_functions.apply_filter(kappa_Q_rec,band_pass_filter)
        kappa_RS_sum_with_bias_smoothed=map_making_functions.apply_filter(kappa_RS_sum_with_bias,band_pass_filter)
        kappa_sum_with_bias_smoothed=map_making_functions.apply_filter(kappa_sum_with_bias,band_pass_filter)

   
        plt.figure(1); 
        plt.rc('text', fontsize=6)
        plt.subplot(2,2,1); plt.imshow(conv_map,cmap='RdBu'); plt.colorbar(); plt.title('$\kappa_{IN}$');  
        plt.subplot(2,2,2); plt.imshow(kappa_Q_rec,cmap='RdBu'); plt.colorbar() ; plt.title('$\kappa_{QE}$'); 
        plt.subplot(2,2,3); plt.imshow(kappa_RS_sum_with_bias,cmap='RdBu'); plt.colorbar(); plt.title('$\kappa_{RS,CONV}$'); 
        plt.subplot(2,2,4); plt.imshow(kappa_sum_with_bias,cmap='RdBu'); plt.colorbar(); plt.title('$\kappa_{RS,FFT}$'); 


		
        plt.figure(2);  plt.rc('text', fontsize=6)
        plt.subplot(2,2,1); plt.imshow(conv_map_smoothed,cmap='RdBu'); plt.colorbar(); plt.title('$\kappa_{IN}$');  
        plt.subplot(2,2,2); plt.imshow(kappa_Q_rec_smoothed,cmap='RdBu'); plt.colorbar() ; plt.title('$\kappa_{QE}$'); 
        plt.subplot(2,2,3); plt.imshow(kappa_RS_sum_with_bias_smoothed,cmap='RdBu'); plt.colorbar(); plt.title('$\kappa_{RS,CONV}$'); 
        plt.subplot(2,2,4); plt.imshow(kappa_sum_with_bias_smoothed,cmap='RdBu'); plt.colorbar(); plt.title('$\kappa_{RS,FFT}$'); 
        plt.show()  
     		

        raise KeyboardInterrupt

        #kappa_noise=estimators.real_space_estimator2(just_noise_T,just_noise_T, ideal_filter)
        #kappa_noise_sum+=kappa_noise
        #kappa_unlensed_sum+=bias

    # FOR OTHER MAP ESTIMATORS BELOW HERE NEED TO IMPLEMENT REAL SPACE *MAP-BASED* VERSION  <<<<<<<<<< KAVI
	#np.savetxt(working_dir+'/LensingField/'+est+'_RS_kappa_plus_bias_out_'+ spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_iteration'+str(i),np.real(kappa_RS_sum_with_bias))

    

	np.savetxt(working_dir+'/LensingField/'+est+'_RS_kappa_plus_bias_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_iteration'+str(i)+'_noisefreeunlensedmap',np.real(kappa_RS_sum_with_bias))
	np.savetxt(working_dir+'/LensingField/'+est+'_RS_bias_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_iteration'+str(i)+'_noisefreeunlensedmap',np.real(bias_RS))
	np.savetxt(working_dir+'/LensingField/'+est+'_RS_kappa_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_iteration'+str(i),np.real(kappa_RS))
	
    raise KeyboardInterrupt

    if spec=='ee':
        kappa=estimators.real_space_estimator(E_lensed, ideal_filter)
        bias=estimators.real_space_estimator(E, ideal_filter)
        
        kappa-=bias  #This is not possible for actual cmb maps
        #kappa_noise=estimators.real_space_estimator(just_noise_E, ideal_filter)
        kappa_sum+=kappa

    if spec=='te':
        """
        kappa_1=estimators.real_space_estimator2(T_lensed,E_lensed, ideal_filter)
        #bias_1=estimators.real_space_estimator2(T, E, ideal_filter)
        kappa_1-=bias_1
        kappa_1_sum+=kappa_1
        
        kappa_2=estimators.real_space_estimator2(E_lensed,T_lensed, ideal_filter)
        #bias_2=estimators.real_space_estimator2(E, T, ideal_filter)
        kappa_2-=bias_2
        kappa_2_sum+=kappa_2
        
        kappa=(kappa_1+kappa_2)/2
        """
        kappa=estimators.real_space_estimator2(T_lensed,E_lensed, ideal_filter)
        #kappa_noise=estimators.real_space_estimator2(just_noise_T,just_noise_E, ideal_filter)
        bias=estimators.real_space_estimator2(T, E, ideal_filter)
        kappa-=bias  #This is not possible for actual cmb maps
        kappa_sum+=kappa

    if spec=='tb':
        kappa=estimators.real_space_estimator2(B_lensed,T_lensed, ideal_filter)
        bias=estimators.real_space_estimator2(B, T, ideal_filter)
        kappa-=bias
        #kappa_noise=estimators.real_space_estimator2(just_noise_T,just_noise_B, ideal_filter)
        kappa_sum+=kappa

    if spec=='eb':
        kappa=estimators.real_space_estimator2(E_lensed,B_lensed, ideal_filter)
        bias=estimators.real_space_estimator2(E, B, ideal_filter)
        #kappa=estimators.real_space_estimator2(B_lensed,E_lensed, ideal_filter)
        #bias=estimators.real_space_estimator2(B, E, ideal_filter)
        kappa-=bias
        #kappa_noise=estimators.real_space_estimator2(just_noise_E,just_noise_B, ideal_filter)
        kappa_sum+=kappa

    if spec=='plus':
        kappa1=estimators.real_space_estimator(Q_lensed, ideal_filter)
        kappa2=estimators.real_space_estimator(U_lensed, ideal_filter)
        kappa=kappa1+kappa2
        bias1=estimators.real_space_estimator(Q, ideal_filter)
        bias2=estimators.real_space_estimator(U, ideal_filter)
        bias=bias1+bias2
        kappa-=bias
        kappa_sum+=kappa

    if spec=='minus':
        kappa1=estimators.real_space_estimator(Q_lensed, ideal_filter)
        kappa2=estimators.real_space_estimator(U_lensed, ideal_filter)
        kappa=kappa1-kappa2
        bias1=estimators.real_space_estimator(Q, ideal_filter)
        bias2=estimators.real_space_estimator(U, ideal_filter)
        bias=bias1-bias2
        kappa-=bias
        kappa_sum+=kappa

    if spec=='cross':
        filtQ=np.loadtxt(working_dir+'/Filters/'+exp_filt+'_te_shear_plus_filt_'+str(fov_deg)+'_'+str(nside))
        filtU=np.loadtxt(working_dir+'/Filters/'+exp_filt+'_te_shear_cross_filt_'+str(fov_deg)+'_'+str(nside))
        kappa1=estimators.real_space_estimator2(Q_lensed, T_lensed, filtQ)
        kappa2=estimators.real_space_estimator2(U_lensed, T_lensed, filtU)
        kappa=kappa1+kappa2
        bias1=estimators.real_space_estimator2(Q, T, filtQ)
        bias2=estimators.real_space_estimator2(U, T, filtU)
        bias=bias1+bias2
        kappa-=bias
        kappa_sum+=kappa


    print 'bias:', bias
    print 'kappa:',kappa

    np.savetxt(working_dir+'/LensingField/'+est+'_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_iteration'+str(i),np.real(kappa))
    if use_noisy_unlensed:
        np.savetxt(working_dir+'/LensingField/'+est+'_bias_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_iteration'+str(i),np.real(bias))
    else:
        np.savetxt(working_dir+'/LensingField/'+est+'_bias_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_iteration'+str(i)+'_noisefreeunlensedmap',np.real(bias))

		
## >>> UPDATE OUTPUT FILE NAMES ABOVE

#  exit
#sort out names - add spectra and noise details
#np.savetxt(working_dir+'/Reconstructions/conv_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_sum/num_maps)

#np.savetxt(working_dir+'/Reconstructions/conv_from_noise_'+str(fov_deg)+'_'+str(nside),kappa_noise)
#np.savetxt(working_dir+'/Reconstructions/conv_from_unlensed_'+str(fov_deg)+'_'+str(nside),kappa_unlensed)
i=0
if(use_noisy_maps):
    T_lensed=np.loadtxt(working_dir+'/Maps/T_lensed_with_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
        
    ##E_lensed=np.loadtxt(working_dir+'/Maps/E_lensed_with_QU_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
    ##B_lensed=np.loadtxt(working_dir+'/Maps/B_lensed_with_QU_noise_'+exp+'_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
else:   #just used lensed map, no noise added
    T_lensed=np.loadtxt(working_dir+'/Maps/T_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
    ##E_lensed=np.loadtxt(working_dir+'/Maps/E_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))
    ##B_lensed=np.loadtxt(working_dir+'/Maps/B_lensed_'+str(i)+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(fudge))

	## >>> UPDATE INPUT FILE NAMES ABOVE

if spec=='tt':
    kappa_1_map=estimators.real_space_estimator(T_lensed, ideal_filter)
    np.savetxt(working_dir+'/LensingField/'+est+'_from_1_map_'+spec+'_'+str(fov_deg),kappa_1_map)
    #now using just one lensed map, and making many approximations to unlensed map by randomising phases
    for j in range(num_rand_phase_maps):
        T_rand_phase=map_making_functions.scramble_phases(T_lensed)
        bias_rand_phase=estimators.real_space_estimator(T_rand_phase, ideal_filter)
        np.savetxt(working_dir+'/LensingField/'+est+'_bias_from_rand_phases_'+str(j)+'_'+spec+'_'+str(fov_deg),np.real(bias_rand_phase))
if spec=='eb':
    kappa_1_map=estimators.real_space_estimator2(E_lensed, B_lensed, ideal_filter)
    np.savetxt(working_dir+'/LensingField/'+est+'_from_1_map_'+spec+'_'+str(fov_deg),kappa_1_map)
    for j in range(num_rand_phase_maps):
        E_rand_phase=map_making_functions.scramble_phases(E_lensed)
        B_rand_phase=map_making_functions.scramble_phases(B_lensed)
        bias_rand_phase=estimators.real_space_estimator2(E_rand_phase, B_rand_phase, ideal_filter)
        np.savetxt(working_dir+'/LensingField/'+est+'_bias_from_rand_phases_'+str(j)+'_'+spec+'_'+str(fov_deg),np.real(bias_rand_phase))

## >>> UPDATE OUTPUT FILE NAMES ABOVE &&& IMPLEMENT THIS FOR OTHER MAP COMBINATIONS


print('Making band pass filter')
band_pass_filter=map_making_functions.make_filter(band_pass_fun, fov_deg, nside)
band_pass_filter_RS_size=map_making_functions.make_filter(band_pass_fun, fov_deg*(nside-2*nside/8+1)/float(nside), nside-2*nside/8+1)



"""
plt.subplot(131)
plt.imshow(bias)
plt.colorbar(shrink=0.5, extend='both')
plt.title('Reconstruction from unlensed')
plt.subplot(132)
plt.imshow(kappa)
plt.colorbar(shrink=0.5, extend='both')
plt.title('Reconstruction from lensed maps')
plt.subplot(133)
plt.imshow(kappa-bias)
plt.colorbar(shrink=0.5, extend='both')
plt.title('Difference')
plt.show()

plt.subplot(131)
plt.imshow(map_making_functions.apply_filter(bias,band_pass_filter))
plt.colorbar(shrink=0.5, extend='both')
plt.title('Reconstruction from unlensed')
plt.subplot(132)
plt.imshow(map_making_functions.apply_filter(kappa,band_pass_filter))
plt.colorbar(shrink=0.5, extend='both')
plt.title('Reconstruction from lensed maps')
plt.subplot(133)
plt.imshow(map_making_functions.apply_filter((kappa-bias),band_pass_filter))
plt.colorbar(shrink=0.5, extend='both')
plt.title('Difference')

plt.show()
"""

## >>> UPDATE OUTPUT FILE NAMES BELOW

if est=='conv':
    conv_in_smoothed=map_making_functions.apply_filter(conv_map,band_pass_filter)
    np.savetxt(working_dir+'/LensingField/conv_in_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside),conv_map)
    
    if(use_noisy_maps):
        np.savetxt(working_dir+'/LensingField/conv_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_sum/num_maps)

    
        np.savetxt(working_dir+'/LensingField/conv_out_with_bias_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_sum_with_bias/num_maps)

        np.savetxt(working_dir+'/LensingField/conv_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_no_bias',kappa)
    else:
        np.savetxt(working_dir+'/LensingField/conv_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'noisefreemaps',kappa_sum/num_maps)
        
        np.savetxt(working_dir+'/LensingField/conv_out_with_bias_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'noisefreemaps',kappa_sum_with_bias/num_maps)
        
        np.savetxt(working_dir+'/LensingField/conv_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_no_bias_noisefreemaps',kappa)



    #np.savetxt(working_dir+'/LensingField/conv_from_noise_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_noise_sum/num_maps)
    #np.savetxt(working_dir+'/LensingField/conv_from_unlensed_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_unlensed_sum/num_maps)

elif est=='shear_plus':
    conv_in_smoothed=map_making_functions.apply_filter(shear_map_plus,band_pass_filter)
    np.savetxt(working_dir+'/LensingField/shear_plus_in_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside),np.real(shear_map_plus))
    
    if(use_noisy_maps):
        np.savetxt(working_dir+'/LensingField/shear_plus_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_sum/num_maps)
        #np.savetxt(working_dir+'/LensingField/shear_plus_from_noise_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_noise)
        #np.savetxt(working_dir+'/LensingField/shear_plus_from_unlensed_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_unlensed)
        
        np.savetxt(working_dir+'/LensingField/shear_plus_out_with_bias_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_sum_with_bias/num_maps)

        np.savetxt(working_dir+'/LensingField/shear_plus_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_no_bias',kappa)
    else:
        np.savetxt(working_dir+'/LensingField/shear_plus_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'noisefreemaps',kappa_sum/num_maps)
        #np.savetxt(working_dir+'/LensingField/shear_plus_from_noise_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_noise)
        #np.savetxt(working_dir+'/LensingField/shear_plus_from_unlensed_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_unlensed)
        
        np.savetxt(working_dir+'/LensingField/shear_plus_out_with_bias_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'noisefreemaps',kappa_sum_with_bias/num_maps)
        
        np.savetxt(working_dir+'/LensingField/shear_plus_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_no_bias_noisefreemaps',kappa)

#get rid of num_maps stuff
elif est=='shear_cross':
    conv_in_smoothed=map_making_functions.apply_filter(shear_map_cross,band_pass_filter)
    np.savetxt(working_dir+'/LensingField/shear_cross_in_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside),np.real(shear_map_cross))   #check this
    if(use_noisy_maps):
        np.savetxt(working_dir+'/LensingField/shear_cross_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_sum/num_maps)
        #np.savetxt(working_dir+'/LensingField/shear_cross_from_noise_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_noise)
        #np.savetxt(working_dir+'/LensingField/shear_cross_from_unlensed_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_unlensed)
        
        np.savetxt(working_dir+'/LensingField/shear_cross_out_with_bias_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_sum_with_bias/num_maps)

        np.savetxt(working_dir+'/LensingField/shear_cross_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_no_bias',kappa)
    else:
        np.savetxt(working_dir+'/LensingField/shear_cross_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'noisefreemaps',kappa_sum/num_maps)
        #np.savetxt(working_dir+'/LensingField/shear_cross_from_noise_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_noise)
        #np.savetxt(working_dir+'/LensingField/shear_cross_from_unlensed_'+spec+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'maps',kappa_unlensed)
        
        np.savetxt(working_dir+'/LensingField/shear_cross_out_with_bias_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_'+str(num_maps)+'noisefreemaps',kappa_sum_with_bias/num_maps)
        
        np.savetxt(working_dir+'/LensingField/shear_cross_out_'+spec+'_'+exp+'_'+str(fov_deg)+'_'+str(nside)+'_no_bias_noisefreemaps',kappa)


kappa_smoothed=map_making_functions.apply_filter(kappa_sum/num_maps,band_pass_filter)
kappa_1_map_bias_subtracted_smoothed=map_making_functions.apply_filter(kappa,band_pass_filter)

kappa_1_map_no_bias_subtracted_smoothed=map_making_functions.apply_filter(kappa+bias,band_pass_filter)

#kappa_smoothed_rand_phase=map_making_functions.apply_filter(kappa_sum_rand_phase/num_rand_phase_maps,band_pass_filter)
kappa_smoothed_RS=map_making_functions.apply_filter(kappa_RS_sum/num_maps,band_pass_filter)#_RS_size)



#kappa_noise_smoothed=map_making_functions.apply_filter(kappa_noise_sum/num_maps,band_pass_filter)
#kappa_unlensed_smoothed=map_making_functions.apply_filter(kappa_unlensed_sum/num_maps,band_pass_filter)


"""
plt.imshow(kappa_smoothed,aspect='auto',origin='lower', extent=[0,40,0,40], cmap='RdBu')
#plt.colorbar()
actual=plt.contour(conv_in_smoothed,linewidths=2, aspect='auto',origin='lower', extent=[0,40,0,40], colors="black")#, cmap='RdBu')#colors="black"
#plt.clabel(actual, inline=1, fontsize=10)
#plt.colorbar(actual, shrink=0.8, extend='both')
plt.title('Real Space Lensing Field Reconstruction')
plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps.png')
plt.show()
"""
if est=='conv':
    max=0.02
else:
    max=0.015
kappa_smoothed_rescaled=kappa_smoothed*np.sqrt(2*np.pi)#*nside**4/(fov_deg*(pi/180.))**2
kappa_1_map_bias_subtracted_smoothed_rescaled=kappa_1_map_bias_subtracted_smoothed*np.sqrt(2*np.pi)
kappa_1_map_no_bias_subtracted_smoothed_rescaled=kappa_1_map_no_bias_subtracted_smoothed*np.sqrt(2*np.pi)
conv_in_smoothed_rescaled=conv_in_smoothed
#plt.imshow(kappa_smoothed_rescaled,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)#norm=norm)
#plt.colorbar(shrink=0.8, extend='both')
#actual=plt.contour(conv_in_smoothed_rescaled,linewidths=2, aspect='auto',origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)# norm=norm)#colors="black"
##plt.clabel(actual, inline=1, fontsize=10)
#plt.colorbar(actual, shrink=0.8, extend='both')
#plt.title(spec.upper()+' Real Space '+est_name+' Reconstruction minus bias')
#if(use_noisy_maps):
#    plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps.png')
#else:
#    plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps_noisefreemap.png')
#plt.show()


#plt.subplot(121)
#plt.imshow(kappa_1_map_bias_subtracted_smoothed_rescaled,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)
#plt.colorbar(shrink=0.5, extend='both')
#plt.title('With bias subtracted')
#plt.subplot(122)
#plt.imshow(map_making_functions.apply_filter(kappa+bias,band_pass_filter)*np.sqrt(2*np.pi),origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)
#plt.colorbar(shrink=0.5, extend='both')
#plt.title('No bias subtraction')
#plt.show()


#plt.imshow(kappa_1_map_no_bias_subtracted_smoothed,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)#norm=norm)
#plt.colorbar(shrink=0.8, extend='both')
#actual=plt.contour(conv_in_smoothed_rescaled,linewidths=2, aspect='auto',origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)# norm=norm)#colors="black"
##plt.clabel(actual, inline=1, fontsize=10)
#plt.colorbar(actual, shrink=0.8, extend='both')
#plt.title(spec.upper()+' Real Space '+est_name+' Reconstruction')
#if(use_noisy_maps):
#    plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps_with_bias.png')
#else:
#    plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps_with_bias_noisefreemap.png')
#plt.show()
#
#plt.imshow(kappa_1_map_bias_subtracted_smoothed,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)#norm=norm)
#plt.colorbar(shrink=0.8, extend='both')
#actual=plt.contour(conv_in_smoothed_rescaled,linewidths=2, aspect='auto',origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)# norm=norm)#colors="black"
##plt.clabel(actual, inline=1, fontsize=10)
#plt.colorbar(actual, shrink=0.8, extend='both')
#plt.title(spec.upper()+' Real Space '+est_name+' Reconstruction')
#if(use_noisy_maps):
#    plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps_with_bias.png')
#else:
#    plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps_with_bias_noisefreemap.png')
#plt.show()

"""
plt.imshow(kappa_smoothed_rand_phase,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)#norm=norm)
plt.colorbar(shrink=0.8, extend='both')
actual=plt.contour(conv_in_smoothed_rescaled,linewidths=2, aspect='auto',origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)# norm=norm)#colors="black"
#plt.clabel(actual, inline=1, fontsize=10)
plt.colorbar(actual, shrink=0.8, extend='both')
plt.title(spec.upper()+' Real Space Lensing Field Reconstruction with Random Phases')
plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps_rescaled_colouredcontours.png')
plt.show()
"""

#plt.imshow(kappa_smoothed_rescaled, origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)# norm=norm)
#plt.colorbar()
#actual=plt.contour(conv_in_smoothed,linewidths=2, origin='lower', extent=[0,fov_deg,0,fov_deg], colors="black", norm=norm)
##plt.clabel(actual, inline=1, fontsize=10)
##plt.colorbar(actual, shrink=0.8, extend='both')
#plt.title(spec.upper()+' Real Space '+est_name+' Reconstruction '+str(num_maps)+' Maps')
#plt.show()
##plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps_rescaled.png')
##plt.show()
#
#plt.imshow(kappa_1_map_bias_subtracted_smoothed_rescaled, origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)#, norm=norm)
#plt.colorbar()
#actual=plt.contour(conv_in_smoothed,linewidths=2, origin='lower', extent=[0,fov_deg,0,fov_deg], colors="black", norm=norm)
##plt.clabel(actual, inline=1, fontsize=10)
##plt.colorbar(actual, shrink=0.8, extend='both')
#plt.title(spec.upper()+' Real Space '+est_name+' Reconstruction')# 1 Map')
#plt.show()
plt.figure()

plt.imshow(map_making_functions.apply_filter(kappa+bias,band_pass_filter)*np.sqrt(2*np.pi), origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu', vmin=-max, vmax=max)# norm=norm)
plt.colorbar()
actual=plt.contour(conv_in_smoothed,linewidths=2, origin='lower', extent=[0,fov_deg,0,fov_deg], colors="black", norm=norm)
#plt.clabel(actual, inline=1, fontsize=10)
#plt.colorbar(actual, shrink=0.8, extend='both')
plt.title(spec.upper()+' Real Space '+est_name+' Reconstruction')# 1 Map')
#plt.title(spec.upper()+' Real Space Shear Plus Reconstruction')
plt.savefig('/Users/Heather/Dropbox/MSc/PlotCode/Reconstructions/'+exp+'/'+spec.upper()+'_'+est+'_1map')
plt.show()





#fov_RS_size=fov_deg#*(nside-2*nside/8+1)/float(nside)
#conv_in_smoothed_RS_size=conv_in_smoothed#[nside/8:nside-nside/8+1, nside/8:nside-nside/8+1]
#kappa_smoothed_RS_rescaled=kappa_smoothed_RS#*nside**4/(fov_deg*(pi/180.))**2
#plt.imshow(kappa_smoothed_RS_rescaled, origin='lower', extent=[0,fov_RS_size,0,fov_RS_size], cmap='RdBu', norm=norm)
#plt.colorbar(shrink=0.8, extend='both')
#actual=plt.contour(conv_in_smoothed_RS_size,linewidths=2, aspect='auto',origin='lower', extent=[0,fov_RS_size,0,fov_RS_size], cmap='RdBu', norm=norm)#colors="black"
##plt.clabel(actual, inline=1, fontsize=10)
#plt.colorbar(actual, shrink=0.8, extend='both')
#plt.title(spec.upper()+' Real Space Lensing Field Reconstruction with full convolution')
##plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps_conv_in_RS.png')
##plt.show()

#plt.imshow(kappa_smoothed_RS_rescaled, origin='lower', extent=[0,fov_RS_size,0,fov_RS_size], cmap='RdBu', norm=norm)
#actual=plt.contour(conv_in_smoothed_RS_size,linewidths=2, aspect='auto',origin='lower', extent=[0,fov_RS_size,0,fov_RS_size], colors="black", norm=norm)
#plt.title(spec.upper()+' Real Space Lensing Field Reconstruction with full convolution')
##plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps_conv_in_RS_black.png')
##plt.show()
#
##plotting same area of harmonic space conv reconstruction for comparison
#plt.imshow(kappa_smoothed_rescaled[nside/8:nside-nside/8+1], origin='lower', extent=[0,fov_RS_size,0,fov_RS_size], cmap='RdBu', norm=norm)
#actual=plt.contour(conv_in_smoothed_RS_size,linewidths=2, aspect='auto',origin='lower', extent=[0,fov_RS_size,0,fov_RS_size], colors="black", norm=norm)
#plt.title(spec.upper()+' Real Space Lensing Field Reconstruction')
##plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'_'+str(int(fov_deg))+'_'+str(nside)+'_'+str(num_maps)+'maps_conv_in_black_RS_size.png')
##plt.show()
#plt.close()

"""
plt.imshow(kappa_noise_smoothed)
plt.contour(conv_in_smoothed,linewidths=2,colors="black")
plt.title('Lensing Field Reconstruction from Noise Map')
plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'filt_'+exp+'noisemaps'+'_noise')
plt.show()

plt.imshow(kappa_unlensed_smoothed)
plt.contour(conv_in_smoothed,linewidths=2,colors="black")
plt.title('Lensing Field Reconstruction from Unlensed Map')
plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'filt_'+exp+'noisemaps'+'_unlensed')
plt.show()

kappa_no_noise_smoothed=kappa_smoothed-kappa_noise_smoothed

plt.imshow(kappa_no_noise_smoothed)
plt.contour(conv_in_smoothed,linewidths=2,colors="black")
plt.title('Lensing Field Reconstruction from Total Map minus Reconstruction from Noise Map')
plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'filt_'+exp+'noisemaps'+'_minus_noise')
plt.show()

kappa_no_noise_no_unlensed_smoothed=kappa_smoothed-kappa_noise_smoothed-kappa_unlensed_smoothed

plt.imshow(kappa_no_noise_no_unlensed_smoothed)
plt.contour(conv_in_smoothed,linewidths=2,colors="black")
plt.title('Lensing Field Reconstruction from Total Map minus Reconstruction from Noise and Unlensed Maps')
plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'filt_'+exp+'noisemaps'+'_minus_noise_minus_unlensed')
plt.show()

kappa_no_unlensed_smoothed=kappa_smoothed-kappa_unlensed_smoothed

plt.imshow(kappa_no_unlensed_smoothed)
plt.contour(conv_in_smoothed,linewidths=2,colors="black")
plt.title('Lensing Field Reconstruction from Total Map minus Reconstruction from Unlensed Map')
plt.savefig(working_dir+'/Reconstructions/'+est+'_'+spec+'_'+exp_filt+'filt_'+exp+'noisemaps'+'_minus_unlensed')
plt.show()
"""




#Explain that diff lensing field used for planck vs act coz dif size maps?? Or fix??






