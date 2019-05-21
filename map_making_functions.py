#combines Jethro's make_map.py, make_polarization_map.py, add_noise.py

import random
import numpy as np
from math import *
import math as M
import pylab
#from IPython.Shell import IPShellEmbed #H commented
import func
import spline
import matplotlib.pyplot as plt
import smooth
import scal_power_spectra
import numpy
import scipy.signal as sg

import inputs
import os

from scipy.interpolate import RectBivariateSpline as spline2d   #is this the right thing to use??

#do in one place


working_dir='/Users/Heather/Documents/masters/polRealSp/Codes/MapsPy/HeatherVersion'

#from make_map
def make_l_array(nside, fov_deg):
   fov_rad=(pi/180.)*fov_deg
   k_max=pi*nside/fov_rad
   l_factor=2*k_max
   k_array=make_k_array(nside)
   l_array=l_factor*k_array
   print l_array.max()
   return(l_array)

#from make_map
def make_lxy_arrays(nside, fov_deg):
   fov_rad=(pi/180.)*fov_deg
   k_max=pi*nside/fov_rad
   l_factor=2*k_max
   kx_array, ky_array=make_kxy_arrays(nside)
   lx_array=l_factor*kx_array
   ly_array=l_factor*ky_array
   return(lx_array,ly_array)

#from make_map   
def make_k_array(nside):
   #def f(x,y):
     #return(sqrt(x**2+y**2))
   #vec_f=np.vectorize(f, otypes=[np.double])
   freq_vec=np.fft.fftfreq(nside)
   kx_array=np.outer(freq_vec,np.ones(nside))
   ky_array=np.outer(np.ones(nside),freq_vec)
   #k_array=vec_f(kx_array, ky_array)
   k_array=np.sqrt(kx_array**2+ky_array**2)
   return(k_array)

#from make_map
def make_kxy_arrays(nside):
   freq_vec=np.fft.fftfreq(nside)
   kx_array=np.outer(freq_vec,np.ones(nside))
   ky_array=np.outer(np.ones(nside),freq_vec)
   print 'ky_array', ky_array
   return(kx_array,ky_array)

#from make_polarization_map
# I think this is algorithm 3 from http://arxiv.org/pdf/1105.2737.pdf
def make_convergence_map(power_func, fov_deg, nside,lx_arr,ly_arr,seed):
    print 'making convergence map'
    my_image=np.zeros(shape=(nside,nside))
    rng=random.Random(seed)
    
    fov_rad=fov_deg*np.pi/180
    DeltaA=(fov_rad/nside)**2
    var=1#/DeltaA
    print var
    for i in range(nside):
        for j in range(nside):
            my_image[i,j]=rng.gauss(0.,var)
    lx_array=lx_arr
    ly_array=ly_arr
   
    #Jet had:
    #fsky=(fov_deg*(pi/180.))**2/(4.*pi)
    norm=1#(1./fsky)*(nside/4.)**2
    
    my_image_fft=np.fft.fft2(my_image)

    for i in range(nside):
        for j in range(nside):
            lx=lx_array[i,j]
            ly=ly_array[i,j]
            l_final=sqrt(lx**2+ly**2)
            factor=power_func(l_final)
            factor=sqrt(factor*norm)
            my_image_fft[i,j]=factor*my_image_fft[i,j]/fov_rad
    my_image_filtered=np.fft.ifft2(my_image_fft)*nside   #normalising by dividing by nside -- is this correct?? should i put it back?
    my_image_filtered=my_image_filtered.real.copy()
    return(my_image_filtered)



#from make_polarization_map
def make_TQU_maps(ps_temp, ps_e1, ps_e2, fov_deg, nside, l_array, phi_l, seed=None):
    print 'making TQU maps'
    
    
    if(seed):
        rng=random.Random(seed)
        for i in range(nside):
            for j in range(nside):
                temp=np.zeros(shape=(nside,nside))#,dtype=complex)
                #t=np.zeros(shape=(nside,nside))#,dtype=complex)
                pol=np.zeros(shape=(nside,nside))#,dtype=complex)
                temp[i,j]=rng.gauss(0.,1.)
                pol[i,j]=rng.gauss(0.,1.)
    else:
        temp=np.random.randn(nside,nside)
        pol=np.random.randn(nside,nside)
    
    
    temp_fft=np.fft.fft2(temp)/nside
    t_fft=temp_fft
    pol_fft=np.fft.fft2(pol)/nside

    fov_rad=fov_deg*np.pi/180
    
    for i in range(nside):
        for j in range(nside):
            factor=ps_temp(l_array[i,j])
            temp_fft[i,j]=factor*temp_fft[i,j]#/fov_rad**2 #these are already square-rooted (tt_func_bis diff from cl_tt_func_bis), H added division by fov**2
            pol_fft[i,j]=t_fft[i,j]*ps_e1(l_array[i,j])+pol_fft[i,j]*ps_e2(l_array[i,j])#/fov_rad**2
    
    Q_fft=pol_fft*np.cos(2*phi_l)
    U_fft=pol_fft*np.sin(2*phi_l)

    T=np.fft.ifft2(temp_fft)*(nside**2)
    Q=np.fft.ifft2(Q_fft)*(nside**2)
    U=np.fft.ifft2(U_fft)*(nside**2)
    print Q[0:10,0:10]
    Q=np.real(Q)
    print Q[0:10,0:10]
    print U[0:10,0:10]
    U=np.real(U)
    print U[0:10,0:10]
    print T[0:10,0:10]
    T=np.real(T)
    print T[0:10,0:10]
    return (T, Q, U)

#creates gaussian random variable in Fourier space
def make_convergence_map_Heather(power_func, fov_deg, nside,lx_arr,ly_arr,seed):
    print 'making convergence map with normalisation from Holmes'
    my_image_fft=np.zeros(shape=(nside,nside),dtype=complex)
    if seed==None:
        rng1=random.Random(seed)
        rng2=random.Random(seed)
    else:
        rng1=random.Random(seed)
        rng2=random.Random(np.sqrt(seed*2))
    
    lx_array=lx_arr
    ly_array=ly_arr
    
    fov_rad=fov_deg*np.pi/180
    norm=fov_rad*sqrt(1/2.)#sqrt(fov_rad**2/2) from miranda holmes, see http://cims.nyu.edu/~holmes/mathcamp/ft_summary.pdf
    
    jay=np.complex(0,1)
    
    for i in range(nside):
        for j in range(nside):
            lx=lx_array[i,j]
            ly=ly_array[i,j]
            l_final=sqrt(lx**2+ly**2)
            factor=sqrt(power_func(l_final))
            my_image_fft[i,j]=norm*factor*(rng1.gauss(0.,1.)+jay*rng2.gauss(0.,1.))
    my_image_filtered=np.fft.ifft2(my_image_fft)/(fov_rad/nside)**2 #dividing by dx**2 to normalise fourier transform, see holmes reference above
    my_image_filtered=my_image_filtered.real.copy()
    return(my_image_filtered)

#creates gaussian random variable in Fourier space
def make_TQU_maps_Heather(ps_temp, ps_e1, ps_e2, fov_deg, nside, l_array, phi_l, seed=None):
    print 'making TQU maps with normalisation from Holmes'
    temp_fft=np.zeros(shape=(nside,nside))
    ##pol_fft=np.zeros(shape=(nside,nside),dtype=complex)
    if seed==None:
        rng1=random.Random(seed)
        rng2=random.Random(seed)
        ##rng3=random.Random(seed)
        ##rng4=random.Random(seed)
    else:
        rng1=random.Random(seed)
        rng2=random.Random(seed*2)
        ##rng3=random.Random(seed*3)
        ##rng4=random.Random(seed*4)

    fov_rad=fov_deg*np.pi/180
    norm=fov_rad*np.sqrt(1/2.)#sqrt(fov_rad**2/2)
    
    jay=np.complex(0,1)
    
    """
    for i in range(nside):
        for j in range(nside):
            temp_fft[i][j]=norm*ps_temp(l_array[i,j])
            ##t=(rng1.gauss(0.,1.)+jay*rng2.gauss(0.,1.))
            ##p=(rng3.gauss(0.,1.)+jay*rng4.gauss(0.,1.))
            ##temp_fft[i,j]=norm*ps_temp(l_array[i,j])*t #these are already square-rooted (tt_func_bis diff from cl_tt_func_bis)
            ##pol_fft[i,j]=norm*(t*ps_e1(l_array[i,j])+p*ps_e2(l_array[i,j]))
    """
    for i in range(nside/2 +1):
        for j in range(i):
            temp_fft[i][j]=norm*ps_temp(l_array[i,j])

    temp_fft[:nside/2+1,:nside/2+1]+=np.transpose(temp_fft[:nside/2+1,:nside/2+1])

    for i in range(1,nside/2+1):
        temp_fft[i][i]=norm*ps_temp(l_array[i,i])

    temp_fft[:,nside/2+1:]=np.flip(temp_fft[:,1:nside/2],1)
    temp_fft[nside/2+1:,:]=np.flip(temp_fft[1:nside/2,:],0)

    temp_fft=np.array(temp_fft,dtype=complex)
    
    temp_fft=temp_fft*(np.random.normal(0.,1.,(nside,nside))+np.random.normal(0.,1.,(nside,nside))*1j)
    
    ##Q_fft=pol_fft*np.cos(2*phi_l)
    ##U_fft=pol_fft*np.sin(2*phi_l)

    T=np.fft.ifft2(temp_fft)/(fov_rad/nside)**2
    ##Q=np.fft.ifft2(Q_fft)/(fov_rad/nside)**2
    ##U=np.fft.ifft2(U_fft)/(fov_rad/nside)**2

    ##Q=np.real(Q)    #real part of gaussian random variable is gaussian random variable
    ##U=np.real(U)
    T=np.real(T)
    ##return (T, Q, U)
    return T
   
#from make_polarization_map
def make_lensing_field(conv_map, nside, fov_deg):
    print 'making lensing field from convergence map'
    jay=0.+1.j
  
    lx,ly=make_lxy_arrays(nside,fov_deg)
    #lx/=2.
    #ly/=2.

    l_arr=make_l_array(nside,fov_deg)
    #l_arr/=2.
    l_sq=l_arr*l_arr
    one_over_l_sq=1./l_sq
    one_over_l_sq[0,0]=1
    lj=one_over_l_sq*jay

    fov_rad=(pi/180.)*fov_deg
    delta_theta=fov_rad/nside

 
    fft_conv_map=np.fft.fft2(conv_map)
    #fft_lensing_potential=lj*fft_conv_map
    fft_lensing_potential=one_over_l_sq*fft_conv_map*(2)   #Heather added *2, changed lj to one_over_l_sq, add negative?? - if I put in a negative then reconstructed convergence is negative
    #check ps of phi - looking good :)
    l_vec,cl_tt, cl_ee, cl_te, cl_tt_lens, cl_ee_lens, cl_bb_lens, cl_te_lens, cl_kk=scal_power_spectra.spectra()
    cl_dd=cl_kk*4 / (l_vec*(l_vec+1))
    cl_pp=cl_dd/(l_vec * (l_vec + 1.0))
    
    """
    l, phi_spec=scal_power_spectra.just_compute_power_spectrum(np.real(np.fft.ifft2(fft_lensing_potential)),fov_deg)
    
    plt.plot(l, np.sqrt(l*(l+1)/(2*pi)*phi_spec), label='from map')
    plt.plot(l_vec, np.sqrt(l_vec*(l_vec+1)/(2*pi)*cl_pp))
    plt.xlim(1, 10000)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('potential power spec')
    plt.show()
    """
    

    fft_x_deflection=lx*jay*fft_lensing_potential #H added jay here and on next line
    fft_y_deflection=ly*jay*fft_lensing_potential

    
    x_deflection=np.real(np.fft.ifft2(fft_x_deflection)) #real
    y_deflection=np.real(np.fft.ifft2(fft_y_deflection))
    
    print 'ave x deflection', np.mean(np.abs(x_deflection))
    print 'ave y deflection', np.mean(np.abs(y_deflection))
    """
    plt.imshow(x_deflection*180/pi*60)
    plt.colorbar()
    plt.title ('x deflection')
    plt.show()
    
    plt.imshow(y_deflection*180/pi*60)
    plt.colorbar()
    plt.title ('y deflection')
    plt.show()
    
    plt.imshow(np.sqrt(x_deflection**2+y_deflection**2)*180/pi*60)
    plt.colorbar()
    plt.title ('deflection angle magnitude')
    plt.show()

    def_mag=np.sqrt(x_deflection**2+y_deflection**2)
    l, def_spec=scal_power_spectra.just_compute_power_spectrum(def_mag,fov_deg)

    l_alpha, c_alpha=scal_power_spectra.compute_deflection_spectrum(x_deflection,y_deflection,fov_deg)
    
    #relate to ps of dx and dy
    
    plt.plot(l_alpha, np.sqrt(l*(l+1)/(2*pi)*c_alpha), label='x and y combined from map')
    #plt.plot(l, np.sqrt(l*(l+1)/(2*pi)*(def_spec_x)), label='y from map')
    #plt.plot(l, np.sqrt(l*(l+1)/(2*pi)*(def_spec_y)), label='x from map')
    plt.plot(l, np.sqrt(l*(l+1)/(2*pi)*def_spec), label='from map')
    plt.plot(l_vec, np.sqrt(l_vec*(l_vec+1)/(2*pi)*cl_dd), label='from camb')
    plt.legend(loc='lower center')
    plt.xlim(1,10000)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('L')
    plt.ylabel(r'$[l(l+1)C^{\alpha\alpha}_L/2\pi]^{1/2}$')
    plt.title('deflection angle power spec')
    plt.show()
    """
    
    
    x_displacement=np.real(np.fft.ifft2(fft_x_deflection))/delta_theta
    y_displacement=np.real(np.fft.ifft2(fft_y_deflection))/delta_theta
    
  
    result=(x_displacement,y_displacement)
    #result=(x_deflection, y_deflection)
    return(result)

   
#from make_map
def distort_map(map_in, displacement_field,nside):
  """
  Inputs: 
    map_in = ndarray of shape=(nside,nside)
          consisting of temperature map
    displacement_field = ndarray of shape=(nside,nside,2)=actual x and y deflections divided by pixel height/width
  Output:
    lensed_map = ndarray of shape=(nside,nside)
  """
  print 'lensing the map'
  lensed_map=np.zeros(shape=(nside,nside))
  i_values=np.outer(np.arange(nside),np.ones(nside))
  j_values=np.outer(np.ones(nside),  np.arange(nside))
  i_values=i_values+displacement_field[:,:,0]
  j_values=j_values+displacement_field[:,:,1]
  
  print('i:', i_values[100:110,100:110])

  
  for i in range(nside):
    for j in range(nside):
      lensed_map[i,j]=lens_map(
          i_values[i,j],j_values[i,j],map_in,nside)

  #plt.imshow(lensed_map-map_in, vmin=-50, vmax=50)
  #plt.colorbar()
  #plt.title('Lensed-unlensed maps from map_making_functions.distort_map')
  #plt.show()
  return(lensed_map)
  
#from make_map, used by distort_map
def lens_map(i,j,map_in,nside):
    i_m=np.int(floor(i)%nside); i_p=np.int(ceil(i)%nside)
    j_m=np.int(floor(j)%nside); j_p=np.int(ceil(j)%nside)
    i_offset=(i-i_m)%nside; j_offset=(j-j_m)%nside
    #print 'Checking:', i_m, i_p, j_m, j_p, i_offset, j_offset
    v_mm=map_in[i_m,j_m]
    v_mp=map_in[i_m,j_p]
    v_pm=map_in[i_p,j_m]
    v_pp=map_in[i_p,j_p]
    value=(
        v_mm*(1.-i_offset)*(1.-j_offset)
       +v_mp*(1.-i_offset)*j_offset
       +v_pm*i_offset*(1-j_offset)
       +v_pp*i_offset*j_offset
             ) #bilinear interpolation to get value at (i,j)
    #print i_offset, j_offset, v_mm, v_mp, v_pm, v_pp, value
    #if i_m%50==1 and j_m%50==1:
        #print 'i j vmm vpp', i, j, v_mm, v_pp, value
    return(value)


def distort_map_high_res(map_in, displacement_field, nside_cmb, nside_lens, fov_deg):
    """
        Inputs:
        map_in = ndarray of shape=(nside,nside)
        consisting of temperature map
        deflection_field = ndarray of shape=(nside,nside,2). Actual x and y deflections in radians
        nside_cmb=number of pixels per side for high-res cmb map (map_in)
        nside_lens=number of pixels per side for lower res deflection field and lensed cmb map
        Output:
        lensed_map = ndarray of shape=(nside,nside)
        """
    print 'lensing the map'
    """
    lensed_map=np.zeros(shape=(nside_lens,nside_lens))
    fov_rad=fov_deg*pi/180
    delta_theta_lens=fov_rad/nside_lens
    x_vec_lens=np.arange(nside_lens)*delta_theta_lens#np.linspace(0, fov_rad, nside_lens)
    y_vec_lens=np.arange(nside_lens)*delta_theta_lens#np.linspace(0, fov_rad, nside_lens)
    x_values=np.outer(x_vec_lens,np.ones(nside_lens))
    y_values=np.outer(np.ones(nside_lens),  y_vec_lens)
                      
    x_values_lensed=x_values+deflection_field[:,:,0]
    y_values_lensed=y_values+deflection_field[:,:,1]
    print('i:', x_values_lensed[100:110,100:110]/delta_theta_lens)

    
    delta_theta_cmb=fov_rad/nside_cmb
    for i in range(nside_lens):
        for j in range(nside_lens):
            lensed_map[i,j]=lens_map(
                                 x_values_lensed[i,j]/delta_theta_cmb,y_values[i,j]/delta_theta_cmb,map_in,nside_cmb)
    """
    
    print 'lensing the map'
    lensed_map=np.zeros(shape=(nside_lens,nside_lens))
    i_values=np.outer(np.arange(nside_lens),np.ones(nside_lens))
    j_values=np.outer(np.ones(nside_lens),  np.arange(nside_lens))
    i_values=i_values+displacement_field[:,:,0]
    j_values=j_values+displacement_field[:,:,1]
            
    #print('i:', i_values[100:110,100:110])
    hi_to_lo=int(nside_cmb/float(nside_lens))
    #print 'conversion factor', hi_to_lo
            
    for i in range(nside_lens):
        for j in range(nside_lens):
            lensed_map[i,j]=lens_map(i_values[i,j]*hi_to_lo,j_values[i,j]*hi_to_lo,map_in,nside_cmb)
    """
    if nside_lens!=nside_cmb:
        hi_to_lo_i_vec=np.arange(0, nside_cmb, hi_to_lo, dtype=int)
        hi_to_lo_j_vec=np.arange(0, nside_cmb, hi_to_lo, dtype=int)
        #print 'should be equal:', hi_to_lo_i_vec.shape, nside_lens
        hi_to_lo_i=np.outer(hi_to_lo_i_vec,np.ones(nside_lens)).astype(int)
        hi_to_lo_j=np.outer(np.ones(nside_lens), hi_to_lo_j_vec).astype(int)
        lo_res_map=map_in[hi_to_lo_i, hi_to_lo_j]
    else:
        lo_res_map=map_in
    
    plt.imshow(lensed_map-lo_res_map, vmin=-50, vmax=50)
    plt.colorbar()
    plt.title('Lensed-unlensed maps from map_making_functions.distort_map_Heather')
    plt.show()"""
    return(lensed_map)


  
#from add_noise
def add_noise(map_in,fov_deg,nside,theta, delta, seed):
    # Heather added instance of numpy.random.RandomState with seed to make noise predictable for certain tests to be run
    # From COrE white paper 100+143+217
    print 'Adding noise in map_making_functions.add_noise'
    fwhm_arcmin=theta
    sigma=(M.pi/(180.*60.))*fwhm_arcmin/M.sqrt(8.*M.log(2.))
    # Noise vector in muK*arcmin for 14 month survey
    n=delta
    def cmb_weight_func(l):
      b=M.exp(-0.5*(sigma*l)**2)
      s=(n/b)**2
      cmb_weight  =(n**2/b)/s
      return(cmb_weight)
    def noise_weight_func(l):
      b=M.exp(-0.5*(sigma*l)**2)
      s=(n/b)**2
      s2=n**4/b**2
      noise_factor=M.sqrt(s2/s)
      return(noise_factor)
    map_in_f=np.fft.fft2(map_in)
    prng=np.random.RandomState(seed) #pseudorandom number generator
    # Following line generates a noise map of 1. muK*arcmin
    noise_map=1e-6*prng.randn(nside**2)/((60.*fov_deg)/nside)
    noise_map.shape=(nside,nside)
    l_array=make_l_array(nside,fov_deg)
    noise_map_f=np.fft.fft2(noise_map)
    del noise_map
    cmb_weight_map  =np.vectorize(  cmb_weight_func)(l_array) 
    noise_weight_map=np.vectorize(noise_weight_func)(l_array)
    map_out_f=cmb_weight_map*map_in_f+noise_weight_map*noise_map_f
    map_out=np.real(np.fft.ifft2(map_out_f))
    plt.figure()
    plt.plot(l_array, noise_weight_map)
    plt.title('in add_noise in map_making_functions.py')
    #plt.show()
    return(map_out)
    
#from make_map, used in reconstruct_shear...
def make_filter(filter_function, fov_deg, nside):
    filter=np.zeros(shape=(nside,nside),dtype=complex)
    l_array=make_l_array(nside, fov_deg)
    filter_fun_vec=np.vectorize(filter_function,otypes=[np.double])
    filter=filter_fun_vec(l_array)
    return(filter)

#from make_map, used in reconstruct_shear...
def apply_filter(map_in, filter):   #filter is in Fourier space
   map_in_fft=np.fft.fft2(map_in)
   map_fft_filtered=map_in_fft*filter
   map_out=np.fft.ifft2(map_fft_filtered)
   map_out=(map_out.real).copy()
   return(map_out)

#convolution in real space - set ncut to nside/4 (gives ok reconstruction, nside/2 is better - check) and boundary to wrap 
def apply_RS_filter(map_in , filter_RS):
    nside=filter_RS.shape[0]
    ncut=nside/4 # nside/4 # nside/10  # ncut = nside/2 gives full RS filter -> nside/4 is ok
    filter_RS_cut = filter_RS[nside/2-ncut:nside/2+ncut, nside/2-ncut:nside/2+ncut]
    #plt.imshow(np.log(np.abs(filter_RS_cut)))
    #plt.title('In map_making_functions, filter should be zero outside of displayed region')
    #plt.colorbar()
    #plt.show()
    #raise KeyboardInterrupt
    #map_filtered= sg.fftconvolve(map_in,filter_RS,mode='same')
    map_filtered= sg.convolve2d(map_in,filter_RS_cut,mode='same', boundary='wrap') # 'fill') # boundary='wrap')
    print'filtered map shape', map_filtered.shape
    return(map_filtered)

#from make_map, used in reconstruct_shear...
def conv_to_shear(conv_map,nside):
  conv_map_f=np.fft.fft2(conv_map)
  kx, ky = make_kxy_arrays(nside)
  ksq=kx**2+ky**2
  zero_ind=0
  ksq[zero_ind,zero_ind]=1.0
  """plus_mask=(kx**2-ky**2)/ksq
  cross_mask=2*kx*ky/ksq
  plus_f    =plus_mask*conv_map_f
  cross_f   =cross_mask*conv_map_f
  plus_map = np.fft.ifft2(plus_f )
  cross_map= np.fft.ifft2(cross_f)
  """
  plus_map = np.fft.ifft2((kx**2-ky**2)/ksq*conv_map_f)
  cross_map= np.fft.ifft2(2*kx*ky/ksq*conv_map_f)
  """
  phi1=conv_map_f//ksq
  phi2=plus_f/(kx**2-ky**2)
  phi3=cross_f/(2*kx*ky)
  
  plt.imshow(np.fft.ifft2(phi1).astype('float'))
  plt.colorbar()
  plt.show()
  plt.imshow(np.fft.ifft2(phi2).astype('float'))
  plt.colorbar()
  plt.show()
  plt.imshow(np.fft.ifft2(phi3).astype('float'))
  plt.colorbar()
  plt.show()"""
  plus_map=plus_map.real.copy() ; cross_map=cross_map.real.copy()
  result=(plus_map,cross_map)
  return(result)

def shear_plus_cross_to_E_B(shear_plus, shear_cross, nside, fov_deg):
    f_plus=np.fft.fft2(shear_plus)
    f_cross=np.fft.fft2(shear_cross)
    """
    plt.imshow(np.real(shear_plus))
    plt.colorbar()
    plt.show()
    plt.imshow(np.real(shear_cross))
    plt.colorbar()
    plt.show()"""
    lx_array, ly_array=make_lxy_arrays(nside,fov_deg)
    cos_2phi_l=(lx_array**2-ly_array**2)/(lx_array**2+ly_array**2)
    sin_2phi_l=(2*lx_array*ly_array)/(lx_array**2+ly_array**2)
    cos_2phi_l[0,0]=0#1
    sin_2phi_l[0,0]=0#1
    """
    plt.imshow(cos_2phi_l)
    plt.show()
    plt.imshow(sin_2phi_l)
    plt.show()"""
    f_E=cos_2phi_l*f_plus+sin_2phi_l*f_cross
    f_B=-1*sin_2phi_l*f_plus+cos_2phi_l*f_cross
    shear_E=np.real(np.fft.ifft2(f_E))
    shear_B=np.real(np.fft.ifft2(f_B))
    """
    print shear_E
    print shear_B
    plt.imshow(shear_E)
    plt.title('shear E mode in map_making_functions.shear_plus_cross_to_E_B')
    plt.colorbar()
    plt.show()
    plt.imshow(shear_B)
    plt.title('shear B mode in map_making_functions.shear_plus_cross_to_E_B')
    plt.colorbar()
    plt.show()"""
    return(shear_E, shear_B)

def scramble_phases(map):   #Heather added to try approximate unlensed map as lensed with scrambled phases to remove bias in power spec
    nside=map.shape[0]
    map_fft=np.fft.fft2(map)
    magnitude=np.absolute(map_fft)
    rand_phases=np.random.random_sample(map.shape)*2*pi
    #make hermite symmetric to get real map - phase=0 for real number and phase changes sign for conjugate
    rand_phases[0,0]=rand_phases[nside/2,0]=rand_phases[0,nside/2]=rand_phases[nside/2,nside/2]=0.
    rand_phases[nside/2, nside/2+1:nside]=-1*(np.flipud(rand_phases[nside/2, 1:nside/2]))
    rand_phases[0, nside/2+1:nside]=-1*(np.flipud(rand_phases[0, 1:nside/2]))
    rand_phases[nside/2+1:nside,0]=-1*(np.flipud(rand_phases[1:nside/2,0]))
    rand_phases[nside/2+1:nside, 1:nside]=-1*(np.fliplr(np.flipud(rand_phases[1:nside/2, 1:nside])))
    rand_phases=np.fft.fftshift(rand_phases)
    
    
    map_fft=magnitude*np.exp(1j*rand_phases)
    return(np.real(np.fft.ifft2(map_fft)))

