import numpy as np
import h5py,time, matplotlib.pyplot as plt
from scipy.optimize import fmin_cg, minimize
from drift.core import manager
import corrcal2
import sys
sys.path.insert(0,'/home/zahra/PIPELINE')
from log_red_cal_new import Vis_only, Bls_counts, colour_scatterplot, Scatterplot
from decimal import Decimal
from operator import add,sub
from scipy.optimize import LinearConstraint, BFGS, Bounds
import scipy as sp

ops=(add,sub)


#bt=manager.beamtransfer.BeamTransfer('/home/zahra/PIPELINE/ex_3by3_kl/bt_matrices/bt/')
#mkl=manager.kltransform.KLTransform(bt)
#sig_cov_mat,noise_cov_mat=mkl.sn_covariance(1)

'''
why does noise not affect gain recovery? I saved gain fits for sigma=10^{-2} with vecs=1000,0(redundant case) and gain fluc=1.e-6 with initial guess random(and equal for each run)
of 1.e-7, with 1000 runs, and got the difference between these gain fits and the gain fits for different noise and signal. The difference was zero.
'''

'''
ts_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/rand_gains/amp_pt001_phase_pt001/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/rand_gains/sig_pt0012.h5','r')
gn_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis/rand_gains/ts_final/app_gain_noise_2.h5','r')

ts_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_no_gain_fluc/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_no_gain_fluc/rand_gains/amp_low_phase_low/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_no_gain_fluc/rand_gains/sig_low2.h5','r')
gn_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_no_gain_fluc/rand_gains/ts_final/app_gain_noise_2.h5','r')


ts_2=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_many_noise/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_many_noise/rand_gains/amp_phase/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_many_noise/rand_gains/sig_2.h5','r')
gn_2_50_1=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_many_noise/rand_gains/ts_final_50K_1day/app_gain_noise_2.h5','r')
gn_2_500_1=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_many_noise/rand_gains/ts_final_500K_1day/app_gain_noise_2.h5','r')
gn_2_750_1=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_many_noise/rand_gains/ts_final_750K_1day/app_gain_noise_2.h5','r')
gn_2_5e4_1=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_many_noise/rand_gains/ts_final_5e4K_1day/app_gain_noise_2.h5','r')

'''
ts_2=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_pt01/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_pt01/rand_gains/amp_phase/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_pt01/rand_gains/sig_2.h5','r')
gn_2_50_1=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_pt01/rand_gains/ts_final_50K_1day/app_gain_noise_2.h5','r')
'''

ts_2=h5py.File('/home/zahra/PIPELINE/ex_7by7_hex/draco_synthesis_many_noise/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/ex_7by7_hex/draco_synthesis_many_noise/rand_gains/amp_phase/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/ex_7by7_hex/draco_synthesis_many_noise/rand_gains/sig_2.h5','r')
gn_2_50_1=h5py.File('/home/zahra/PIPELINE/ex_7by7_hex/draco_synthesis_many_noise/rand_gains/ts_final_50K_1day/app_gain_noise_2.h5','r')
#gn_2_500_1=h5py.File('/home/zahra/PIPELINE/ex_7by7_hex/draco_synthesis_many_noise/rand_gains/ts_final_500K_1day/app_gain_noise_2.h5','r')
#gn_2_750_1=h5py.File('/home/zahra/PIPELINE/ex_7by7_hex/draco_synthesis_many_noise/rand_gains/ts_final_750K_1day/app_gain_noise_2.h5','r')
#gn_2_5e4_1=h5py.File('/home/zahra/PIPELINE/ex_7by7_hex/draco_synthesis_many_noise/rand_gains/ts_final_5e4K_1day/app_gain_noise_2.h5','r')



ts_2=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_pt1_gainfluc/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_pt1_gainfluc/rand_gains/amp_pt1_phase_pt1/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_pt1_gainfluc/rand_gains/sig_pt12.h5','r')
gn_2=h5py.File('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/draco_synthesis_pt1_gainfluc/rand_gains/ts_final/app_gain_noise_2.h5','r')


ts_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt01_gain_fluc/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt01_gain_fluc/rand_gains/amp_pt01_phase_pt01/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt01_gain_fluc/rand_gains/sig_pt012.h5','r')
gn_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt01_gain_fluc/rand_gains/ts_final/app_gain_noise_2.h5','r')

ts_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt1_gain_fluc/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt1_gain_fluc/rand_gains/amp_pt1_phase_pt1/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt1_gain_fluc/rand_gains/sig_pt12.h5','r')
gn_2=h5py.File('/home/zahra/PIPELINE/ex_3by3_kl/draco_synthesis_pt1_gain_fluc/rand_gains/ts_final/app_gain_noise_2.h5','r')
'''

#m=manager.ProductManager.from_config('/home/zahra/PIPELINE/ex_3by3_kl/prod_params_custom.yaml')
m=manager.ProductManager.from_config('/home/zahra/PIPELINE/example_7by7_lmax_mmax_250/prod_params_custom.yaml')
#m=manager.ProductManager.from_config('/home/zahra/PIPELINE/ex_7by7_hex/prod_params_custom.yaml')

Scatterplot(m)

t=m.telescope
x=t.feedpositions[:,0] #these are x and y positions not x and y polarizations
y=t.feedpositions[:,1]


#klobj=m.kltransforms['kl']

time_channel=400
Ndish=49
correlation_arr,sum_counts,corr_counts=Bls_counts(m)

Nbls,_=correlation_arr.shape
arr_pt01=Vis_only(m,ts_2,rg_2,ag_2,gn_2_50_1,time_channel,50,1)
#arr_pt1=Visibilities_grid(m,ts_2,rg_2,ag_2,gn_2_500_1,time_channel,Tsys=500,ndays=1)
#arr_pt5=Visibilities_grid(m,ts_2,rg_2,ag_2,gn_2_750_1,time_channel,Tsys=750,ndays=1)
#arr_10=Visibilities_grid(m,ts_2,rg_2,ag_2,gn_2_5e4_1,time_channel,Tsys=5e4,ndays=1)

arr=arr_pt01
vis=arr[0][:,time_channel]

#2 is gnoise, 0 is manual
sigma=arr[-1] #sigma_gn
#sigma=1


time_arr=np.arange(325,335,1)



def Vis_arr_real(Nbls,time_arr):
    vis_arr=np.zeros((Nbls,time_arr.size))
    for i in range(Nbls):
        for j in range(len(time_arr)):
            int_time_arr=np.int(time_arr[j])
            vis_arr[np.int(i),np.int(j)]=arr[2][i,int_time_arr].real
    return np.mean(vis_arr,axis=1)



def Vis_arr_imag(Nbls,time_arr):
    vis_arr=np.zeros((Nbls,time_arr.size))
    for i in range(Nbls):
        for j in range(len(time_arr)):
            int_time_arr=np.int(time_arr[j])
            vis_arr[np.int(i),np.int(j)]=arr[2][i,int_time_arr].imag
    return np.mean(vis_arr,axis=1)


'''
plt.plot(time_arr,Vis_arr(0))
plt.plot(time_arr,Vis_arr(1))
plt.plot(time_arr,Vis_arr(7))
plt.plot(time_arr,Vis_arr(10))
plt.plot(time_arr,Vis_arr(20))
plt.plot(time_arr,Vis_arr(30))
plt.xlabel('time channels')
plt.ylabel('|Visibilities|')
plt.title('Visibilities as a function of time for a single baseline')
plt.show()
'''
#sigma=1.e-6
#data=np.append(vis.real,vis.imag)
#data=np.ones(72)



#data[0::2]=Vis_arr_real(Nbls,time_arr)
#data[1::2]=Vis_arr_imag(Nbls,time_arr)

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

vecs=np.load('sky_cov_mat_stacked_real_imaginary_shape_2by72.npy')
vecs_mat_sky=np.append(vecs[0][:vis.size],vecs[1][:vis.size])
vecs_mat_sky=np.diag(vecs_mat_sky)

vec_real=1000*np.load('vec_real.npy')
vec_imag=1000*np.load('vec_imag.npy')
#vecs=np.array([vec_real,vec_imag])
#print (np.array([vec_real,vec_imag]).shape)

v1=np.zeros(2*vis.size)
v1[0::2]=1
v2=np.zeros(2*vis.size)
v2[1::2]=1
vecs=1.e3*np.vstack([v1,v2])

v1_1d=v1[:vis.size]
v2_1d=v2[:vis.size]
vecs_1d=1000*np.append(v1_1d,v2_1d)
vecs_mat=np.diag(vecs_1d)

#vecs_mat=np.random.uniform(0,1.,2*vis.size)

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
diag_mat=np.diag(diag)
diag=diag.reshape(1,2*vis.size)

src=np.zeros(2*vis.size)
#src=1000*v1

ant1=correlation_arr[:,0].astype(int)
ant2=correlation_arr[:,1].astype(int)

gain=rg_2['gain'][0,:Ndish,time_channel]
sim_gains=np.append(gain.real,gain.imag)

sim_gains=np.zeros(Ndish*2)
sim_gains[0::2]=gain.real
sim_gains[1::2]=gain.imag

sim_gains_amp=sim_gains[0::2]
sim_gains_phase=sim_gains[1::2]

'''
gvec=np.array([])
for i in range(len(sim_gains)):
    gvec=np.append(gvec,np.random.normal(0,.01,2*Ndish)[i]+sim_gains[i])
'''

random=np.random.normal(0,1.e-1,2*Ndish)
#random=np.random.uniform(1.e-7,9.e-7,2*Ndish)
#np.save('random_1e-7_9e-7_uniform',random)
#print (random)

#random=np.load('random_pt1.npy')
mult=random+1.

gvec=np.array([])
for i in range(len(sim_gains)):
    gvec=np.append(gvec,sim_gains[i]+random[i])
    #gvec=np.append(gvec,sim_gains[i]*mult[i])

'''
gvec=np.array([])
for i in range(len(sim_gains)):
    gvec=np.append(gvec,sim_gains[i]*mult[i])
    #gvec=np.append(gvec,np.random.choice(ops)(sim_gains[i],random[i]*sim_gains[i]))
'''
#print (sim_gains)
#print (random)
#print (gvec)
#gvec=np.zeros(len(sim_gains))
#gvec[0::2]=1.

runs=1000
'''
runs=1
gvec=np.zeros((runs,2*Ndish))
for j in range(runs):
    for i in range(len(sim_gains)):
        gvec[j,i]=np.random.normal(0,.1,2*Ndish)[i]+sim_gains[i]
#gvec=np.append(rand_gains.real,rand_gains.imag)
'''
#print (gvec,'gvec')
mat=corrcal2.sparse_2level(diag,vecs,src,2*lims) #init

'''
mat_ones=mat*np.ones(len(data))
mycov=mat.copy() #copy, init
app_gains=mycov.apply_gains_to_mat(gvec,ant1,ant2) #nothing
#print (np.linalg.pinv(mycov*np.ones((len(data)/2,2))),'my own inv')
mycov_inv=mycov.inv() #inv, copy, init
#print (mycov_inv*np.ones((len(data),len(data))),'mycov_inv')
sd=mycov_inv*data #ans because you're multiplying
chisq=np.sum(sd*data) #nothing
nn=gvec.size/2 #nothing
#print (chisq)
normfac=1.
chisq=chisq+normfac*( (np.sum(gvec[1::2]))**2 + (np.sum(gvec[0::2])-nn)**2) #nothing
print (chisq)




args_dense=(data,diag_mat,vecs_mat,ant1,ant2)
#x=np.linspace(-.98,1.2,18)
chi_sq=corrcal2.get_chisq(gvec,*args)
chi_sq_dense=corrcal2.get_chisq_dense(gvec,*args_dense)
'''




gg=np.zeros((runs,Ndish*2))


for run in range(runs):
    arr_pt01=Vis_only(m,ts_2,rg_2,ag_2,gn_2_50_1,time_channel,5000,1)
    arr=arr_pt01
    vis=arr[0][:,time_channel]
    data=np.zeros(2*vis.size)
    data[0::2]=vis.real
    data[1::2]=vis.imag
    fac=1.;
    normfac=1.
    asdf=fmin_cg(corrcal2.get_chisq,gvec*fac,corrcal2.get_gradient,(data,mat,ant1,ant2,fac,normfac))
    #asdf=fmin_cg(corrcal2.get_chisq_dense,gvec*fac,corrcal2.get_gradient_dense,(data,diag_mat,vecs_mat,ant1,ant2,fac,normfac))
    fit_gains=asdf/fac
    gg[run,:]=fit_gains
    #fit_gains_complex=fit_gains[0::2]+np.complex(0,1)*fit_gains[1::2]
    #gg_complex=fit_gains_complex/np.abs(np.mean(fit_gains_complex))
    #gg[m,:][0::2]=gg_complex.real
    #gg[m,:][1::2]=gg_complex.imag


        #fit_gains[m,:]=fit_gains[m,:]/(np.mean(abs_full))

            #print (np.complex(fit_gains[m,i],fit_gains[m,1+i]),'comp fit gains')
    #for i in range(Ndish):
        #fit_gains[m,:]=fit_gains[m,:]/(np.mean(np.abs(np.complex(fit_gains[m,i],fit_gains[m,9+i]))))

#np.save('fit_gains_4.npy',fit_gains)

print (gg.shape,'gg shape')
#print (np.mean(gg_dish0_amp), 'gain mean dish 0') -this gives the same value as gain_mean[0]
gg_amp=gg[:,0::2]
gg_amp_mean=np.mean(gg_amp,axis=1) #shape is number of runs
gg_amp_std=np.std(gg_amp,axis=1)

plt.hist(gg_amp_std,'auto',color='g')
plt.xlabel('Standard deviation')
plt.show()

cent_dish_hex_amp=38
corner_dish_hex_amp=0

cent_dish_sq_amp=24 #16,17,18,30,31,32,23,25
corner_dish_sq_amp=0 #6,42,48

def hist_indiv_dishes_amp(cent_dish,corner_dish):
    gg_dish_cent_amp=gg_amp[:,cent_dish]
    gg_dish_corner_amp=gg_amp[:,corner_dish]

    rel_err_dish_corner_amp=np.abs(gg_dish_corner_amp-sim_gains_amp[corner_dish])/sim_gains_amp[corner_dish]
    rel_err_dish_cent_amp=np.abs(gg_dish_cent_amp-sim_gains[cent_dish])/sim_gains[cent_dish]

    plt.hist(rel_err_dish_corner_amp,'auto',color='g')
    plt.hist(rel_err_dish_cent_amp,'auto',color='r')
    plt.legend(('Outer','Centre'))
    plt.xlabel('Relative error')
    plt.show()

hist_indiv_dishes_amp(cent_dish_sq_amp,corner_dish_sq_amp)

gain_std=(np.std(gg,axis=0)/np.sqrt(runs)).flatten()
gain_mean=np.mean(gg,axis=0).flatten() #shape is 98, we take every second one for amp so shape is 49

print (gvec[0::2],'gvec')
#print (gain_mean[0::2]-gain_mean_sig_pt01[0::2],'gain mean diff')
print (sim_gains,'sim gains')

#print (np.abs(gain_mean[0::2]-sim_gains[0::2]).min(),np.abs(gain_mean[0::2]-sim_gains[0::2]).max(),'diff gains')
#print (gain_std[0::2],'gain std')
#print (gain_std[0::2]-gain_std_sig_pt01[0::2],'gain std diff')


rec_gains_amp=gain_mean[0::2]
rec_gains_phase=gain_mean[1::2]
gain_std_amp=gain_std[0::2]
gain_std_phase=gain_std[1::2]

rel_err=np.abs(rec_gains_amp-sim_gains_amp)/sim_gains_amp




print (gvec[0::2]-gain_mean[0::2],'diff gvec and gain fit')



print (np.sum(rec_gains_amp))
print (gain_std,'gain std')
print (rec_gains_amp.min(),rec_gains_amp.max(), 'gain mean amp max and min')
print (rec_gains_phase.min(),rec_gains_phase.max(), 'gain mean phase max and min')

#print (rel_err,'rel err')
#print (rel_err_1_run,'rel err 1 run')
print ((np.abs(rel_err)).min(),(np.abs(rel_err)).max(), 'rel err max and min')
#print (rel_err_1_run.min(),rel_err_1_run.max(), 'rel err max and min 1 run')

print (gain_mean,'gain mean')
print (np.mean(rec_gains_amp),'mean of amp')
print (np.std(rec_gains_amp),'std of amp')

#sim_gains_amp=sim_gains[:Ndish]
#rec_gains_amp=gain_mean[:Ndish]
#sim_gains_phase=sim_gains[Ndish:]
#rec_gains_phase=gain_mean[Ndish:]
#gain_std_amp=gain_std[:Ndish]
#gain_std_phase=gain_std[Ndish:]

#np.save('3by3_gain_mean_1e5_runs_signal_1000_redundant_gainfluc_1e-6_gain_input_1e-7_scalefac_1',gain_mean)
#np.save('7by7_gain_std_sig_1e-2_runs_1_signal_1000_redundant_gainfluc_1e-6_gain_input_1e-5_times_gainfluc',gain_std)
#colors=['#E49B0F','#007F66','#6082B6','#AB92B3','#00AB66','#A57C00','#D4AF37','#FFD700','#E6BE8A','#85754E','#996515','#A8E4A0','#00FF00','#1CAC78','#008000','#66B032','#1164B4','#2887C8','#A99A86','#2a3439','#5218FA','#E9D66B','#FF7A00','#DF73FF','#F400A1','#006DB0','#49796B','#71A6D2','#319177','#ED2939','#B2EC5D','#4B0082','#FF4F00','#BA160C','#B3446C','#F08080','#FFA07A','#B0C4DE','#AE98AA','#C19A6B','#9F4576','#AAF0D1','#F653A6','#D0417E','#FF0090','#FDBE02','#FF8243','#880085','#915F6D','#0A7E8C','#FEBAAD','#0A7E8C','#997A8D','#AD4379']


colour_scatterplot(m,rel_err)

'''
fig, ax = plt.subplots()

lims = [-1.5,  # min of both axes
    1+np.max([ax.get_xlim()]),  # max of both axes
]
ax.scatter(sim_gains_amp,rec_gains_amp)#,c=colors[:Ndish])
plt.errorbar(sim_gains_amp,rec_gains_amp,xerr=None,yerr=gain_std_amp,linestyle="None")

ax.plot(lims, lims,'g')
#plt.errorbar(x_sim[:Ndish,time_channel],x_rec[:Ndish,time_channel],xerr=None,yerr=error_single[:Ndish],linestyle="None")
plt.xlabel('Simulated gain amplitude')
plt.ylabel('Recovered gain amplitude')
#plt.ylim(.95,1.05)
#plt.xlim(.95,1.05)

plt.show()



gain_mean_0=gain_mean[0]
chisq_0=corrcal2.get_chisq(gain_mean,*args)
print (gain_mean,'gain mean')

args=(data,mat,ant1,ant2)


gain_arr=np.linspace(.8,1.3,100.)

corrcal_arr=np.array([])
for i in gain_arr:
    print (i)
    gain_mean[0]=i
    corrcal_sing=corrcal2.get_chisq(gain_mean,*args)
    corrcal_arr=np.append(corrcal_arr,corrcal_sing)


plt.plot(gain_arr,corrcal_arr)
plt.plot(gain_mean_0,chisq_0,'o')
plt.xlabel('gain amplitude')
plt.ylabel(r'$\rm{\chi^2}$')
plt.show()
#gain_mean_sig_pt01=np.load('7by7_gain_mean_sig_1e-2_runs_1_signal_1000_redundant_gainfluc_1e-6_gain_input_1e-7_scalefac_1.npy')
#gain_std_sig_pt01=np.load('7by7_gain_std_sig_1e-2_runs_1000_signal_1000_redundant_gainfluc_1e-6_gain_input_1e-5_times_gainfluc.npy')
#print (gvec.flatten()[0::2],'sim gains with fluctuation')
#print (fit_gains.flatten()[0::2],'fit gains')

#gain_mean_1_run=np.load('3by3_gain_mean_1_run_signal_1000_redundant_gainfluc_1e-6_gain_input_1e-7_scalefac_1.npy')
'''
