#run this first to make the maps you will perform the reconstructions on

#combines Jethro's 'make_clean_polarization_maps.py' and 'polarization_maps_with_noise.py'

#This code takes nside (number of pixels per side in map) and fov_deg (field of view in degrees) as command-line arguments. Combination of nside and fov_deg gives resolution
#NB! Change working_dir variable in this code and in map_making_functions.py
#make sure your working directory contains a folder called Maps and a folder called LensingField

# Run : import sys ; sys.argv=['make_maps_with_noise.py','2048','20'] ; execfile('make_maps_with_noise.py')

import numpy as np
from math import *
import pylab
#from IPython.Shell import IPShellEmbed #Heather commented
import matplotlib.pyplot as plt
import scal_power_spectra
import map_making_functions
import sys
import time

import inputs
import os

#set these:
make_maps=True   #set to False if you want to use maps you already have saved, and just add different noise to them, set to True if you want to start from scratch with making the maps
add_noise_to_maps=True
num_maps=1
exp='advact' # 'planck'  #'planck' or 'ref'

# USING nside = 2048 and 20 deg - takes about 15-20 mins to run and about 5 GB of RAM

#Code i added
sys.argv=['make_maps_with_noise.py','1024','20']

args=sys.argv
# Kavi use 2048 (nside for kernel, output lensed cmb, lensing field; nside for unlensed is higher - 4096) and 20 deg in all codes
# some values that we have been using are nside=1024, fov=20 (so nside of unlensed cmb maps we use to do the lensing is 4096). I think this is on the edge of what we need to simulate lensing properly
nside_lens=int(args[1])
fov_deg=float(args[2])

if nside_lens<=1024:
    nside_cmb=nside_lens*4
else:
    nside_cmb=4096


print 'CMB map pixel width is', fov_deg/nside_cmb*60,'arcminutes'
print 'Deflection field and lensed map pixel width is', fov_deg/nside_lens*60,'arcminutes'


#modify working_dir to be the directory where you want the code to save everything, and make sure it contains folders called Maps, Filters, LensingField and  Reconstructions

fudge = inputs.fudge # fudge=1.    #fudge factor for deflection field - in inputs.py now

if (exp[0:3]=='ref'):
    working_dir=inputs.working_dir_base + 'ref' + '/fov'+ str(int(fov_deg))
    if not os.path.exists(working_dir):
		os.makedirs(working_dir)
else:
    working_dir=inputs.working_dir_base + exp +'/fov'+str(int(fov_deg))     #+'/jetnormconv'
    if not os.path.exists(working_dir):
		os.makedirs(working_dir)

working_dir_Maps=inputs.working_dir_base+exp+'/fov'+str(int(fov_deg))+'/Maps'
if not os.path.exists(working_dir_Maps):
	os.makedirs(working_dir_Maps)

working_dir_LensingField=inputs.working_dir_base+exp+'/fov'+str(int(fov_deg))+'/LensingField'
if not os.path.exists(working_dir_LensingField):
	os.makedirs(working_dir_LensingField)
		
#we can choose whether we want the convergence map and the noise to be different every time we run the program (set seed to none) or the same each time (set seed to a number)
seedConv=253 #just because I like the number 253. It doesn't matter so long as the lensing field stays the same. The T, Q and U seeds must be random for multiple realizations of the CMB though.
#parameters for noise



#Planck noise
delta_t_am_planck=27.
delta_p_am_planck=40*np.sqrt(2)
sigma_am_planck= 7.#15.

#Reference experiment 
# NOTE - THIS WAS NOT UPDATED FOR RECONSTRUCTIONS on MATHEW MAPS as MAPS SUPPLIED NOT GENERATED HERE
sigma_am_ref=1.4#60*fov_deg/nside   #understand beam stuff
delta_t_am_ref=2.
delta_p_am_ref=np.sqrt(2)*delta_t_am_ref

if exp=='planck':
    delta_T=delta_t_am_planck
    delta_P=delta_p_am_planck
    theta=sigma_am_planck

if exp=='act':
    delta_T=15
    delta_P=delta_T*np.sqrt(2)
    theta=1.5

if exp=='ref':
    delta_T=delta_t_am_ref
    delta_P=delta_p_am_ref
    theta=sigma_am_ref

if exp=='advact':   #150 GHz channel from Henderson et al
    delta_T=7
    delta_P=delta_T*np.sqrt(2)
    theta=1.4


elif (exp[0:3]=='ref'):
    delta_T=float(exp[3:])
    delta_P=delta_T*np.sqrt(2)
    theta=15#sigma_am_ref

exp_print = exp + '_noise_'+str(round(delta_T,2))+'_'+'beam_'+str(round(theta,2))
map_print =  '_'+str(nside_lens)+'_'+str(int(fov_deg))

l_array_lens=map_making_functions.make_l_array(nside_lens, fov_deg)
lx_lens,ly_lens=map_making_functions.make_lxy_arrays(nside_lens,fov_deg)

phi_l_lens=np.arctan(ly_lens/lx_lens)
phi_l_lens[0,0]=0.
phi_l_lens=-phi_l_lens


l_array_cmb=map_making_functions.make_l_array(nside_cmb, fov_deg)
lx_cmb,ly_cmb=map_making_functions.make_lxy_arrays(nside_cmb,fov_deg)

phi_l_cmb=np.arctan(ly_cmb/lx_cmb)
phi_l_cmb[0,0]=0.
phi_l_cmb=-phi_l_cmb

print(sadasd)

if make_maps:
    conv_map=map_making_functions.make_convergence_map_Heather(scal_power_spectra.cl_kk_func_bis, fov_deg, nside_lens,lx_lens,ly_lens, seed=seedConv)

    np.savetxt(working_dir+'/Maps/conv_in'+map_print,conv_map)
    plt.figure()
    plt.imshow(conv_map)
    plt.title('Convergence Map')
    plt.savefig(working_dir+'/Maps/conv_in'+map_print+'.png')
    plt.show()
    
    shear_plus_map, shear_cross_map = map_making_functions.conv_to_shear(conv_map,nside_lens)

    np.savetxt(working_dir+'/Maps/shear_plus_in'+map_print,shear_plus_map)
    plt.figure()
    plt.imshow(shear_plus_map)
    plt.title('Shear Plus Map')
    plt.savefig(working_dir+'/Maps/shear_plus_in'+map_print+'.png')
    plt.show()
	
    np.savetxt(working_dir+'/Maps/shear_cross_in'+map_print,shear_cross_map)
    plt.figure()
    plt.imshow(shear_cross_map)
    plt.title('Shear Cross Map')
    plt.savefig(working_dir+'/Maps/shear_cross_in'+map_print+'.png')
    plt.show()
	
    deflection_field=np.zeros(shape=(nside_lens,nside_lens,2))
    d_x,d_y=map_making_functions.make_lensing_field(conv_map, nside_lens, fov_deg)
    deflection_field[:,:,0]=d_x
    deflection_field[:,:,1]=d_y

    deflection_field*=fudge
    def_mod_map=np.sqrt(deflection_field[:,:,0]**2+deflection_field[:,:,1]**2)
    np.savetxt(working_dir+'/LensingField/deflection_mod_map'+map_print+'_'+str(fudge),def_mod_map)
    
    plt.figure()
    plt.imshow(def_mod_map)
    plt.title('Deflection Modulus Map')
    plt.savefig(working_dir+'/LensingField/deflection_mod_map'+map_print+'_'+str(fudge)+'.png')
    plt.show()
    

# raise KeyboardInterrupt

#edit to save only the maps we need later to save space!
for i in range(0,num_maps):
    if(make_maps):
        print 'making TQU '+str(i)
        #high res maps
        ###T,Q,U=map_making_functions.make_TQU_maps_Heather(scal_power_spectra.tt_func_bis,scal_power_spectra.e1_func_bis,scal_power_spectra.e2_func_bis, fov_deg, nside_cmb, l_array_cmb,phi_l_cmb)
        t1=time.time()
        T=map_making_functions.make_TQU_maps_Heather(scal_power_spectra.tt_func_bis,scal_power_spectra.e1_func_bis,scal_power_spectra.e2_func_bis, fov_deg, nside_cmb, l_array_cmb,phi_l_cmb)
        t2=time.time()
        print(t2-t1)
        ##Q_fft=np.fft.fft2(Q)
        ##U_fft=np.fft.fft2(U)
        ##E_fft=Q_fft*np.cos(2*phi_l_cmb)+U_fft*np.sin(2*phi_l_cmb)
        ##B_fft=-Q_fft*np.sin(2*phi_l_cmb)+U_fft*np.cos(2*phi_l_cmb)
        ##E=np.fft.ifft2(E_fft)
        ##B=np.fft.ifft2(B_fft)
    
        T=np.real(T) #Heather added
        ##E=np.real(E) #Heather added
        ##B=np.real(B) #Heather added
        ##Q=np.real(Q) #Heather added
        ##U=np.real(U) #Heather added
        """
        np.savetxt(working_dir+'/Maps/T_'+str(i)+'_'+str(fov_deg)+'_'+str(nside_cmb),T)
        np.savetxt(working_dir+'/Maps/Q_'+str(i)+'_'+str(fov_deg)+'_'+str(nside_cmb),Q)
        np.savetxt(working_dir+'/Maps/U_'+str(i)+'_'+str(fov_deg)+'_'+str(nside_cmb),U)
        np.savetxt(working_dir+'/Maps/E_'+str(i)+'_'+str(fov_deg)+'_'+str(nside_cmb),E)
        np.savetxt(working_dir+'/Maps/B_'+str(i)+'_'+str(fov_deg)+'_'+str(nside_cmb),B)
        """
        hi_to_lo=int(nside_cmb/float(nside_lens))
        hi_to_lo_i_vec=np.arange(0, nside_cmb, hi_to_lo, dtype=int)
        hi_to_lo_j_vec=np.arange(0, nside_cmb, hi_to_lo, dtype=int)
        hi_to_lo_i=np.outer(hi_to_lo_i_vec,np.ones(nside_lens)).astype(int)
        hi_to_lo_j=np.outer(np.ones(nside_lens), hi_to_lo_j_vec).astype(int)
        T_lo=T[hi_to_lo_i, hi_to_lo_j]
        ##E_lo=E[hi_to_lo_i, hi_to_lo_j]
        ##B_lo=B[hi_to_lo_i, hi_to_lo_j]
        ##U_lo=U[hi_to_lo_i, hi_to_lo_j]
        ##Q_lo=Q[hi_to_lo_i, hi_to_lo_j]
        np.savetxt(working_dir+'/Maps/T_'+str(i)+map_print,T_lo)
        ##np.savetxt(working_dir+'/Maps/E_'+str(i)+map_print,E_lo)
        ##np.savetxt(working_dir+'/Maps/B_'+str(i)+map_print,B_lo)
        ##np.savetxt(working_dir+'/Maps/U_'+str(i)+map_print,U_lo)
        ##np.savetxt(working_dir+'/Maps/Q_'+str(i)+map_print,Q_lo)
        print 'made and saved unlensed maps'
        print 'lensing maps '+str(i)
        T_lensed=map_making_functions.distort_map_high_res(T, deflection_field, nside_cmb, nside_lens, fov_deg)
        ##Q_lensed=map_making_functions.distort_map_high_res(Q, deflection_field, nside_cmb, nside_lens, fov_deg)
        ##U_lensed=map_making_functions.distort_map_high_res(U, deflection_field, nside_cmb, nside_lens, fov_deg)
        ##Q_lensed_fft=np.fft.fft2(Q_lensed)
        ##U_lensed_fft=np.fft.fft2(U_lensed)


        ##E_lensed_fft=Q_lensed_fft*np.cos(2*phi_l_lens)+U_lensed_fft*np.sin(2*phi_l_lens)
        ##B_lensed_fft=-Q_lensed_fft*np.sin(2*phi_l_lens)+U_lensed_fft*np.cos(2*phi_l_lens)
        ##E_lensed=np.fft.ifft2(E_lensed_fft)
        ##B_lensed=np.fft.ifft2(B_lensed_fft)
    
        ##E_lensed=np.real(E_lensed)  #added by Heather, not sure what it should be
        ##B_lensed=np.real(B_lensed)  #added by Heather
    
        np.savetxt(working_dir+'/Maps/T_lensed_'+str(i)+map_print+'_'+str(fudge),T_lensed)
        ##np.savetxt(working_dir+'/Maps/Q_lensed_'+str(i)+map_print+'_'+str(fudge),Q_lensed)
        ##np.savetxt(working_dir+'/Maps/U_lensed_'+str(i)+map_print+'_'+str(fudge),U_lensed)
        ##np.savetxt(working_dir+'/Maps/E_lensed_'+str(i)+map_print+'_'+str(fudge),E_lensed)
        ##np.savetxt(working_dir+'/Maps/B_lensed_'+str(i)+map_print+'_'+str(fudge),B_lensed)
    
    else: #read in maps you already have 
        T_lensed= np.loadtxt(working_dir+'/Maps/T_lensed_'+str(i)+map_print+'_'+str(fudge))
        #E_lensed= np.loadtxt(working_dir+'/Maps/E_lensed_'+str(i)+map_print+'_'+str(fudge))
        #B_lensed= np.loadtxt(working_dir+'/Maps/B_lensed_'+str(i)+map_print+'_'+str(fudge))
        ##U_lensed= np.loadtxt(working_dir+'/Maps/U_lensed_'+str(i)+map_print+'_'+str(fudge))
        ##Q_lensed= np.loadtxt(working_dir+'/Maps/Q_lensed_'+str(i)+map_print+'_'+str(fudge))
        
        T_lo=np.loadtxt(working_dir+'/Maps/T_'+str(i)+map_print)
        ##Q_lo=np.loadtxt(working_dir+'/Maps/Q_'+str(i)+map_print)
        ##U_lo=np.loadtxt(working_dir+'/Maps/U_'+str(i)+map_print)
        print "Shape T_lensed:", np.shape(T_lensed) #edited by Heather
        
    #adding the noise now
    if(add_noise_to_maps):
        print 'adding noise '+str(i)
        
        seedTnoise=np.random.randint(100000)   #if this is a number, the noise will be the same each time, if you want random noise set to None
        ##seedQnoise=np.random.randint(100000)
        ##seedUnoise=np.random.randint(100000)
        
        T_lensed_with_noise=map_making_functions.add_noise(T_lensed, fov_deg, nside_lens, theta, delta_T, seedTnoise)
        ##U_lensed_with_noise=map_making_functions.add_noise(U_lensed, fov_deg, nside_lens, theta, delta_P, seedUnoise)
        ##Q_lensed_with_noise=map_making_functions.add_noise(Q_lensed, fov_deg, nside_lens, theta, delta_P, seedQnoise)
        
        ##Q_lensed_with_noise_fft=np.fft.fft2(Q_lensed_with_noise)
        ##U_lensed_with_noise_fft=np.fft.fft2(U_lensed_with_noise)
        
        #Previously noise was added directly to E and B maps, but I think it needs to be added to the Q and U maps.
        ##E_lensed_with_QU_noise_fft=Q_lensed_with_noise_fft*np.cos(2*phi_l_lens)+U_lensed_with_noise_fft*np.sin(2*phi_l_lens)
        ##B_lensed_with_QU_noise_fft=-Q_lensed_with_noise_fft*np.sin(2*phi_l_lens)+U_lensed_with_noise_fft*np.cos(2*phi_l_lens)
        ##E_lensed_with_QU_noise=np.fft.ifft2(E_lensed_with_QU_noise_fft)
        ##B_lensed_with_QU_noise=np.fft.ifft2(B_lensed_with_QU_noise_fft)
        
        ##E_lensed_with_QU_noise=np.real(E_lensed_with_QU_noise)  #added by Heather, not sure what it should be
        ##B_lensed_with_QU_noise=np.real(B_lensed_with_QU_noise)  #added by Heather

        
        #add same noise to unlensed maps for bias subtraction, esp for TT
        T_with_noise=map_making_functions.add_noise(T_lo, fov_deg, nside_lens, theta, delta_T, seedTnoise)
        ##U_with_noise=map_making_functions.add_noise(U_lo, fov_deg, nside_lens, theta, delta_P, seedUnoise)
        ##Q_with_noise=map_making_functions.add_noise(Q_lo, fov_deg, nside_lens, theta, delta_P, seedQnoise)
        
        ##Q_with_noise_fft=np.fft.fft2(Q_with_noise)
        ##U_with_noise_fft=np.fft.fft2(U_with_noise)
        

        ##E_with_QU_noise_fft=Q_with_noise_fft*np.cos(2*phi_l_lens)+U_with_noise_fft*np.sin(2*phi_l_lens)
        ##B_with_QU_noise_fft=-Q_lensed_with_noise_fft*np.sin(2*phi_l_lens)+U_with_noise_fft*np.cos(2*phi_l_lens)
        ##E_with_QU_noise=np.fft.ifft2(E_with_QU_noise_fft)
        ##B_with_QU_noise=np.fft.ifft2(B_with_QU_noise_fft)
        
        ##E_with_QU_noise=np.real(E_with_QU_noise)  #added by Heather, not sure what it should be
        ##B_with_QU_noise=np.real(B_with_QU_noise)  #added by Heather
    
        #Heather added to do some checking on noise bias
        #just_noise_T=map_making_functions.add_noise(np.zeros((nside, nside)), fov_deg, nside, theta, delta_T, seedTnoise)
        #change just_noiseE and B maps to come from just_noise_Q and U
        #just_noise_E=map_making_functions.add_noise(np.zeros((nside, nside)), fov_deg, nside, theta, delta_P, seedEnoise)
        #just_noise_B=map_making_functions.add_noise(np.zeros((nside, nside)), fov_deg, nside, theta, delta_P, seedBnoise)
    
        np.savetxt(working_dir+'/Maps/T_lensed_with_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge),T_lensed_with_noise)
        
        ##np.savetxt(working_dir+'/Maps/E_lensed_with_QU_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge),E_lensed_with_QU_noise)
        ##np.savetxt(working_dir+'/Maps/B_lensed_with_QU_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge),B_lensed_with_QU_noise)

        ##np.savetxt(working_dir+'/Maps/Q_lensed_with_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge),Q_lensed_with_noise)
        ##np.savetxt(working_dir+'/Maps/U_lensed_with_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge),U_lensed_with_noise)
        
        np.savetxt(working_dir+'/Maps/T_with_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge),T_with_noise)
        ##np.savetxt(working_dir+'/Maps/E_with_QU_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge),E_with_QU_noise)
        ##np.savetxt(working_dir+'/Maps/B_with_QU_noise_'+exp_print+'_'+str(i)+map_print+'_'+str(fudge),B_with_QU_noise)

        """
        np.savetxt(working_dir+'/Maps/just_noise_T_'+exp+'_'+str(i)+map_print+'_'+str(fudge),just_noise_E)
        print "just noise T saved"
        np.savetxt(working_dir+'/Maps/just_noise_E_'+exp+'_'+str(i)+map_print+'_'+str(fudge),just_noise_E)
        print "just noise E saved"
        np.savetxt(working_dir+'/Maps/just_noise_B_'+exp+'_'+str(i)+map_print+'_'+str(fudge),just_noise_B)
        print "just noise B saved"
        """

#Plots for thesis below
T= np.real(np.loadtxt(working_dir+'/Maps/T_'+str(i)+map_print))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
plt.figure()
plt.imshow(T_lensed,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='Greys_r')
plt.colorbar()
plt.title(r'$\tilde{T}$')
plt.show()

plt.figure()
plt.imshow(T,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='Greys_r')
plt.colorbar()
plt.title(r'${T}$')
plt.show()

plt.figure()
plt.imshow(np.abs(T_lensed-T), origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='Greys_r')
plt.colorbar()
plt.title(r'$|\tilde{T}-T|$')
plt.show()

plt.figure()
plt.imshow(def_mod_map, origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='Greys_r')
plt.colorbar()
plt.title(r'$|\vec{\alpha}|$')
plt.show()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


l_vec,cl_tt, cl_ee, cl_te, cl_tt_lens, cl_ee_lens, cl_bb_lens, cl_te_lens, cl_kk=scal_power_spectra.spectra()
l_fac=l_vec*(l_vec+1)/(2*np.pi)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
plt.figure()
plt.plot(l_vec, l_fac*cl_tt, label='unlensed', color='black')
plt.plot(l_vec, l_fac*cl_tt_lens, label='lensed', color='black', linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,1e4)
plt.ylim(1e-3,1e4)
plt.ylabel(r'$l$')
plt.ylabel(r'$l(l+1)C_l^{TT}/2\pi \,\,[\mu K^2]$')
plt.show()

plt.figure()
plt.plot(l_vec, (cl_tt_lens-cl_tt)/cl_tt, color='black')
plt.ylabel(r'$l$')
plt.ylabel(r'$\Delta C_l^{TT}/C_l^{TT}$')
plt.xlim(1,4000)
plt.ylim(-0.05,0.3)
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
plt.figure()
plt.plot(l_vec, l_fac*cl_ee, label='unlensed', color='black')
plt.plot(l_vec, l_fac*cl_ee_lens, label='lensed', color='black', linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,1e4)
plt.ylim(1e-4,1e2)
plt.ylabel(r'$l$')
plt.ylabel(r'$l(l+1)C_l^{EE}/2\pi \,\,[\mu K^2]$')
plt.show()

plt.figure()
plt.plot(l_vec, (cl_ee_lens-cl_ee)/cl_tt, color='black')
plt.ylabel(r'$l$')
plt.ylabel(r'$\Delta C_l^{EE}/C_l^{EE}$')
plt.xlim(1,4000)
plt.ylim(-0.01,0.02)
plt.show()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#plt.plot(l_vec, l_fac*cl_tt, label='unlensed', color='black')
plt.figure()
plt.plot(l_vec, l_fac*cl_bb_lens, label='lensed', color='black', linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,1e4)
#plt.ylim(1e-3,1e4)
plt.ylabel(r'$l$')
plt.ylabel(r'$l(l+1)C_l^{TT}/2\pi \,\,[\mu K^2]$')
plt.show()



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
plt.figure()
plt.plot(l_vec, cl_kk*4, color='black')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,2e3)
plt.ylim(1e-8,2e-6)
plt.ylabel(r'$l$')
plt.ylabel(r'$l(l+1)C_l^{\alpha}/2\pi$')
plt.show()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#Dalian edited this out
"""
plt.figure()
plt.imshow(U,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu')
plt.title(r'${U}$')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(U_lensed_with_noise, extent=[0,fov_deg,0,fov_deg], cmap='RdBu')
plt.title(r'$\tilde{U}$')
plt.colorbar()
plt.show()
"""


"""
plt.imshow(T_lensed,origin='lower', extent=[0,fov_deg,0,fov_deg])
plt.colorbar()
plt.show()
plt.close()

plt.imshow(T)
plt.colorbar()
plt.show()
plt.close()



plt.imshow(Q_lensed_with_noise, extent=[0,fov_deg,0,fov_deg], cmap='RdBu')
plt.title(r'$\tilde{Q}$')
plt.colorbar()
plt.show()
plt.close()

plt.imshow(Q,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu')
plt.title(r'${Q}$')
plt.colorbar()
plt.show()
plt.close()

plt.imshow(U,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='RdBu')
plt.title(r'${U}$')
plt.colorbar()
plt.show()
plt.close()

plt.imshow(U_lensed_with_noise, extent=[0,fov_deg,0,fov_deg], cmap='RdBu')
plt.title(r'$\tilde{Q}$')
plt.colorbar()
plt.show()
plt.close()

plt.imshow(E,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='Greys_r')
plt.title(r'${E}$')
plt.colorbar()
plt.show()
plt.close()

plt.imshow(E_lensed)
plt.colorbar()
plt.show()
plt.close()

plt.imshow(E_lensed_with_QU_noise)
plt.colorbar()
plt.show()
plt.close()

plt.imshow(B,origin='lower', extent=[0,fov_deg,0,fov_deg], cmap='Greys_r')
plt.title(r'${B}$')
plt.colorbar()
plt.show()
plt.close()

plt.imshow(B_lensed)
plt.colorbar()
plt.show()
plt.close()

plt.imshow(B_lensed_with_QU_noise)
plt.colorbar()
plt.show()
plt.close()
"""