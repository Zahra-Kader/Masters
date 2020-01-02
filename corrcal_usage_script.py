import numpy as np
import h5py,time, matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
from drift.core import manager
import corrcal2
import sys
sys.path.insert(0,'/home/zahra/PIPELINE')
from log_red_cal_new import Visibilities_grid, Bls_counts
#bt=manager.beamtransfer.BeamTransfer('/home/zahra/PIPELINE/ex_3by3_kl/bt_matrices/bt/')
#mkl=manager.kltransform.KLTransform(bt)
#sig_cov_mat,noise_cov_mat=mkl.sn_covariance(1)
ts_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/rand_gains/amp_pt001_phase_pt001/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/rand_gains/sig_pt0012.h5','r')
gn_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/rand_gains/ts_final/app_gain_noise_2.h5','r')

m=manager.ProductManager.from_config('/home/zahra/PIPELINE/ex_3by3_kl/prod_params_custom.yaml')

klobj=m.kltransforms['kl']

time_channel=400
Ndish=9

correlation_arr,sum_counts,corr_counts=Bls_counts(m)
arr=Visibilities_grid(m,ts_2,rg_2,ag_2,gn_2,time_channel)
vis_noise_gains=arr[0][:,time_channel] #2 is gnoise, 0 is manual
sigma=arr[-2] #-1 is sigma_gn, -2 is sigma for manual
#sigma=1.e-6
data=np.append(vis_noise_gains.real,vis_noise_gains.imag)
#data=np.ones(72)

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
vecs=np.append(real_signal,imag_signal)
vecs=vecs.reshape(2,72)
lims=sum_counts
#diag=sigma**2*np.ones(72)/1000
diag=sigma**2*np.ones(72)
src=np.zeros(72)

ant1=correlation_arr[:,0].astype(int)
ant2=correlation_arr[:,1].astype(int)

gain=rg_2['gain'][0,:Ndish,time_channel]
sim_gains=np.append(gain.real,gain.imag)


#gvec=np.array([])
#for i in range(len(sim_gains)):
#    gvec=np.append(gvec,random[i]+sim_gains[i])

runs=1000
gvec=np.zeros((runs,2*Ndish))
for j in range(runs):
    for i in range(len(sim_gains)):
        gvec[j,i]=np.random.normal(0,.001,18)[i]+sim_gains[i]
#gvec=np.append(rand_gains.real,rand_gains.imag)

mat=corrcal2.sparse_2level(diag,vecs,src,lims,isinv=0)

fit_gains=np.zeros((runs,Ndish*2))

for m in range(runs):
    niter=1000;
    fac=1000.0;
    #normfac=1.e-10
    asdf=fmin_cg(corrcal2.get_chisq,gvec[m,:]*fac,corrcal2.get_gradient,(data,mat,ant1,ant2,fac))#,normfac))#,gtol=1.e-10,epsilon=1.e-10)
    fit_gains[m,:]=asdf/fac

    #for i in range(Ndish):
    #    fit_gains[m,:]=fit_gains[m,:]/(np.mean(np.abs(np.complex(fit_gains[m,i],fit_gains[m,9+i]))))

#np.save('fit_gains_4.npy',fit_gains)


gain_std=(np.std(fit_gains,axis=0)/np.sqrt(runs)).flatten()
gain_mean=np.mean(fit_gains,axis=0).flatten()

print (gvec)
print (fit_gains)
print (gain_mean-sim_gains,'diff gains')
#print (sim_gains,'sim gains')
print (gain_std, 'std dev for fit gains')

fig, ax = plt.subplots()

lims = [-1.5,  # min of both axes
    1+np.max([ax.get_xlim()]),  # max of both axes
]
ax.scatter(sim_gains[:Ndish],gain_mean[:Ndish])
plt.errorbar(sim_gains[:Ndish],gain_mean[:Ndish],xerr=None,yerr=gain_std[:Ndish],linestyle="None")

ax.plot(lims, lims,'g')
#plt.errorbar(x_sim[:Ndish,time_channel],x_rec[:Ndish,time_channel],xerr=None,yerr=error_single[:Ndish],linestyle="None")
plt.xlabel('Simulated gains')
plt.ylabel('Recovered gains')
plt.ylim(.95,1.05)
plt.xlim(.95,1.05)

plt.show()
