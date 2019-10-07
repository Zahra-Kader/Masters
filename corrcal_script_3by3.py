import numpy as np
import h5py,time, matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
from drift.core import manager
import corrcal2
import sys
sys.path.insert(0,'/home/zahra/PIPELINE')
from log_red_cal_new import Visibilities_grid, Bls_counts, colour_scatterplot
#bt=manager.beamtransfer.BeamTransfer('/home/zahra/PIPELINE/ex_3by3_kl/bt_matrices/bt/')
#mkl=manager.kltransform.KLTransform(bt)
#sig_cov_mat,noise_cov_mat=mkl.sn_covariance(1)


ts_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/rand_gains/amp_pt001_phase_pt001/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/rand_gains/sig_pt0012.h5','r')
gn_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/rand_gains/ts_final/app_gain_noise_2.h5','r')



ts_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt01_gain_fluc/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt01_gain_fluc/rand_gains/amp_pt01_phase_pt01/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt01_gain_fluc/rand_gains/sig_pt012.h5','r')
gn_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt01_gain_fluc/rand_gains/ts_final/app_gain_noise_2.h5','r')

ts_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt1_gain_fluc/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt1_gain_fluc/rand_gains/amp_pt1_phase_pt1/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt1_gain_fluc/rand_gains/sig_pt12.h5','r')
gn_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt1_gain_fluc/rand_gains/ts_final/app_gain_noise_2.h5','r')


m=manager.ProductManager.from_config('/home/zahra/PIPELINE/ex_3by3_kl/prod_params_custom.yaml')



klobj=m.kltransforms['kl']

time_channel=400
Ndish=9

correlation_arr,sum_counts,corr_counts=Bls_counts(m)
arr=Visibilities_grid(m,ts_2,rg_2,ag_2,gn_2,time_channel)
vis=arr[0][:,time_channel] #2 is gnoise, 0 is manual

diff_vis=arr[0][:,time_channel]-arr[1][:,time_channel]
#plt.plot(diff_vis)
#plt.show()

print (arr[1][:,time_channel].min(), arr[1][:,time_channel].max(),'meas vis no noise')
print (vis.min(), vis.max(),'meas vis noise')
sigma=arr[-2] #-1 is sigma_gn, -2 is sigma for manual
#sigma=1.e-6
data=np.append(vis.real,vis.imag)
#data=np.ones(72)

data=np.zeros(2*vis.size)
data[0::2]=vis.real
data[1::2]=vis.imag

'''
sum_full_real=np.zeros((102,102))
for i in range(201):
    sum_full_real+=m.beamtransfer.project_matrix_sky_to_telescope(i,klobj.signal())[0,:,0,:].real
sum_full_imag=np.zeros((102,102))
for i in range(201):
    sum_full_imag+=m.beamtransfer.project_matrix_sky_to_telescope(i,klobj.signal())[0,:,0,:].imag
sum_real_diag=np.diag(sum_full_real)
sum_imag_diag=np.diag(sum_full_imag)
real_sig_condensed_1=sum_real_diag[3:51][0::4]
real_sig_condensed_2=sum_real_diag[54:102][0::4]

imag_sig_condensed_1=sum_imag_diag[3:51][0::4]
imag_sig_condensed_2=sum_real_diag[54:102][0::4]


real_signal_expanded_1=np.repeat(real_sig_condensed_1,corr_counts)
real_signal_expanded_2=np.repeat(real_sig_condensed_2,corr_counts)

imag_signal_expanded_1=np.repeat(imag_sig_condensed_1,corr_counts)
imag_signal_expanded_2=np.repeat(imag_sig_condensed_2,corr_counts)

real_signal=np.append(real_signal_expanded_1,real_signal_expanded_2)
imag_signal=np.append(imag_signal_expanded_1,imag_signal_expanded_2)

vecs=np.vstack([real_signal,imag_signal])
np.save('sky_cov_mat_stacked_real_imaginary_shape_2by72',vecs)
'''
#vecs=np.load('sky_cov_mat_stacked_real_imaginary_shape_2by72.npy')

vec_real=1000*np.load('vec_real.npy')
vec_imag=1000*np.load('vec_imag.npy')
#vecs=np.array([vec_real,vec_imag])


v1=np.zeros(2*vis.size)
v1[0::2]=1
v2=np.zeros(2*vis.size)
v2[1::2]=1
vecs=1000*np.vstack([v1,v2])

'''
true_vis=arr[1][:,time_channel]

vecs_real=true_vis.real
vecs_imag=true_vis.imag
vecs=np.vstack([vecs_real,vecs_imag])
'''
lims=sum_counts
#lims=np.append(lims,lims)
#diag=sigma**2*np.ones(72)/1000
diag=sigma**2*np.ones(2*vis.size)
src=np.zeros(2*vis.size)

ant1=correlation_arr[:,0].astype(int)
ant2=correlation_arr[:,1].astype(int)

gain=rg_2['gain'][0,:Ndish,time_channel]
sim_gains=np.append(gain.real,gain.imag)

sim_gains=np.zeros(Ndish*2)
sim_gains[0::2]=gain.real
sim_gains[1::2]=gain.imag

'''
gvec=np.array([])
for i in range(len(sim_gains)):
    gvec=np.append(gvec,np.random.normal(0,.01,2*Ndish)[i]+sim_gains[i])
'''

random=np.random.normal(0,.01,2*Ndish)

gvec=np.array([])
for i in range(len(sim_gains)):
    gvec=np.append(gvec,random[i]+sim_gains[i])


runs=100000
'''
runs=1
gvec=np.zeros((runs,2*Ndish))
for j in range(runs):
    for i in range(len(sim_gains)):
        gvec[j,i]=np.random.normal(0,.1,2*Ndish)[i]+sim_gains[i]
#gvec=np.append(rand_gains.real,rand_gains.imag)
'''

mat=corrcal2.sparse_2level(diag,vecs,src,2*lims)



fit_gains=np.zeros((runs,Ndish*2))

#args=(data,mat,ant1,ant2)
#x=np.linspace(-.98,1.2,18)
#print (corrcal2.get_chisq(gvec,*args),'chi sq of x')


for m in range(runs):
    niter=1000;
    fac=1.e9;
    normfac=1.e-8
    asdf=fmin_cg(corrcal2.get_chisq,gvec*fac,corrcal2.get_gradient,(data,mat,ant1,ant2,fac,normfac))
    fit_gains[m,:]=asdf/fac
    abs_full=np.array([])
    for i in range(2*Ndish):
        if i%2==0:
            abs_full=np.append(abs_full,np.abs(np.complex(sim_gains[i],sim_gains[1+i]))) #have no idea what abs value gain to divide by,
                                                                                        #could be absolute fit gains
    #fit_gains[m,:]=fit_gains[m,:]/(np.mean(abs_full))

            #print (np.complex(fit_gains[m,i],fit_gains[m,1+i]),'comp fit gains')
    #for i in range(Ndish):
        #fit_gains[m,:]=fit_gains[m,:]/(np.mean(np.abs(np.complex(fit_gains[m,i],fit_gains[m,9+i]))))

#np.save('fit_gains_4.npy',fit_gains)


gain_std=(np.std(fit_gains,axis=0)/np.sqrt(runs)).flatten()
gain_mean=np.mean(fit_gains,axis=0).flatten()

#print (gvec.flatten()[0::2],'sim gains with fluctuation')
#print (fit_gains.flatten()[0::2],'fit gains')
print (gain_mean[0::2],'gain mean')
print (sim_gains[0::2],'sim gains')
print (np.abs(gain_mean[0::2]-sim_gains[0::2]).min(),np.abs(gain_mean[0::2]-sim_gains[0::2]).max(),'diff gains')
print (gain_std[0::2], 'std dev for fit gains')

sim_gains_amp=sim_gains[0::2]
rec_gains_amp=gain_mean[0::2]
sim_gains_phase=sim_gains[1::2]
rec_gains_phase=gain_mean[1::2]
gain_std_amp=gain_std[0::2]
gain_std_phase=gain_std[1::2]

rel_err=np.abs(rec_gains_amp-sim_gains_amp)/sim_gains_amp
print (rel_err)

#sim_gains_amp=sim_gains[:Ndish]
#rec_gains_amp=gain_mean[:Ndish]
#sim_gains_phase=sim_gains[Ndish:]
#rec_gains_phase=gain_mean[Ndish:]
#gain_std_amp=gain_std[:Ndish]
#gain_std_phase=gain_std[Ndish:]


fig, ax = plt.subplots()

lims = [-1.5,  # min of both axes
    1+np.max([ax.get_xlim()]),  # max of both axes
]
ax.scatter(sim_gains_amp,rec_gains_amp)
plt.errorbar(sim_gains_amp,rec_gains_amp,xerr=None,yerr=gain_std_amp,linestyle="None")

ax.plot(lims, lims,'g')
#plt.errorbar(x_sim[:Ndish,time_channel],x_rec[:Ndish,time_channel],xerr=None,yerr=error_single[:Ndish],linestyle="None")
plt.xlabel('Simulated gains')
plt.ylabel('Recovered gains')
#plt.ylim(.95,1.05)
#plt.xlim(.95,1.05)

plt.show()
