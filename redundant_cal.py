import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.patches as mpatches
from drift.core import manager
from scipy.sparse import lil_matrix


ts_2=h5py.File('/home/zahra/PIPELINE/example_2by3_array/draco_synthesis/maps/tstream_2.h5','r')
ag_2=h5py.File('/home/zahra/PIPELINE/example_2by3_array/draco_synthesis/rand_gains/amp_pt001_phase_pt001/app_gain_2.h5','r')
rg_2=h5py.File('/home/zahra/PIPELINE/example_2by3_array/draco_synthesis/rand_gains/sig_pt0012.h5','r')


ag_2_run2=h5py.File('/home/zahra/PIPELINE/example_2by3_array/draco_synthesis/rand_gains_run2/amp_pt001_phase_pt001/app_gain_2.h5','r')
rg_2_run2=h5py.File('/home/zahra/PIPELINE/example_2by3_array/draco_synthesis/rand_gains_run2/sig_pt0012.h5','r')


ag_2_run3=h5py.File('/home/zahra/PIPELINE/example_2by3_array/draco_synthesis/rand_gains_run3/amp_pt001_phase_pt001/app_gain_2','r')


m = manager.ProductManager.from_config('prod_params.yaml')


def nan_check(variable):
    shape_before = variable.shape
    print (shape_before[0]*shape_before[1])
    shape_after = variable[np.logical_not(np.isnan(variable))].shape
    return shape_after

def prod_ind(ts_file,ind_1,ind_2):
    arr_find=np.where(ts_file==[ind_1, ind_2])[0]
    for i in range(len(arr_find)-1):
        if arr_find[i]==arr_find[i+1]:
            location=arr_find[i]
    return location

def vis(ts_file,b):
    return ts_file['vis'][0,b,:]

def log_of_func(function):
    zer=function==0.
    function[zer]=1.e-10
    return np.log(np.absolute(function))


def Visibilities_grid(u,v,manager_config_file,file_no_gain,file_gain,file_with_gain):
    Ndish=u*v
    Nbls=Ndish*(Ndish-1)/2
    Nfeeds=2*Ndish
    vis_bg=file_no_gain['vis'][0,:,:]
    vis_ag=file_with_gain['vis'][0,:,:]
    rg=file_gain['gain'][0,:,:]
    prods=file_no_gain['index_map']['prod'][:]
    Ncorr,Ntimes=vis_bg.shape

    t=manager_config_file.telescope
    t.feedpositions
    t.baselines

    x=np.ndarray.flatten(t.feedpositions)[::2] #these are x and y positions not x and y polarizations
    y=np.ndarray.flatten(t.feedpositions)[1::2]

    unique=np.unique(t.baselines,axis=0)
    N_unique_bls=len(unique)
    a=unique[:,0]
    b=unique[:,1]

    fig, ax = plt.subplots()
    #ax.scatter(x, y)
    #plt.ylabel('Distance (m)')
    #plt.xlabel('Distance (m)')

    for i in range(0,Ndish):
        j=i+Ndish
        #ax.annotate((i,j),(x[i],y[i]))
    #plt.show()

    correlation_arr=np.array([])
    correlation_indices=np.array([])
    for k in range(0,len(b)):
        for i in range(0,Ndish):
            for j in range(0,Ndish):
                if y[j]-y[i]==b[k]:
                    if x[j]-x[i]==a[k]:
                        arr_sing=i,j,a[k],b[k]
                        correlation_arr=np.append(correlation_arr,arr_sing)
                        correlation_indices=np.append(correlation_indices,prod_ind(prods,i,j))

    correlation_arr=np.reshape(correlation_arr,(Nbls,4))
    corr_unique,corr_counts=np.unique(correlation_arr[:,2:4],return_counts=True,axis=0)
    sum_counts=np.cumsum(corr_counts)


    red_bls=np.split(correlation_indices,sum_counts)
    true_vis_array=np.array([])
    measured_vis_array=np.array([])
    for n in range(len(sum_counts)):
        true_vis_array=np.append(true_vis_array,vis(file_no_gain,red_bls[n][0]))
        for i in red_bls[n]:
            measured_vis_array=np.append(measured_vis_array,vis(file_with_gain,i))
            #plt.plot(vis(file_no_gain,i))
        #plt.show()
    measured_vis_array=np.reshape(measured_vis_array,(Nbls,Ntimes))
    true_vis_array=np.reshape(true_vis_array,(N_unique_bls,Ntimes))
    N_unknowns=Ndish+N_unique_bls
    #A = lil_matrix((Nbls, N_unknowns))
    A=np.zeros((15,13))

    for n in range(Nbls):
        corr_single_i=np.int(correlation_arr[n,0:2][0])
        corr_single_j=np.int(correlation_arr[n,0:2][1])
        A[n,corr_single_i]=1
        A[n,corr_single_j]=1

    sum_counts_new=np.append(np.array([0]),sum_counts)
    for i in range(len(sum_counts_new)-1):
        A[sum_counts_new[i]:sum_counts_new[i+1],Ndish+i]=1
    #A=A.tocsr()
    gain=np.array([])
    for i in range(0,Ndish):
        gain_ind=rg[i]
        gain=np.append(gain,gain_ind)
    gain=np.reshape(gain,(Ndish,Ntimes))
    log_gain_amp=log_of_func(gain.real)
    log_gain_phase=log_of_func(gain.imag)
    log_mv_real=log_of_func(measured_vis_array.real)
    log_mv_imag=log_of_func(measured_vis_array.imag)
    log_tv_real=log_of_func(true_vis_array.real)
    log_tv_imag=log_of_func(true_vis_array.imag)
    x_true_real=np.vstack((log_gain_amp,log_tv_real))
    x_true_imag=np.vstack((log_gain_phase,log_tv_imag))
    x_rec_real=np.zeros((N_unknowns,Ntimes))
    x_rec_imag=np.zeros((N_unknowns,Ntimes))


    rows,cols=log_mv_real.shape
    linear_constr_arr=np.append(np.ones(Ndish),np.zeros(N_unique_bls))
    for i in range(cols):

        #bounds=sp.optimize.Bounds([gain_lb,gain_lb,gain_lb,gain_lb,gain_lb,gain_lb,-100.,-100.,-100.,-100.,-100.],
                                  #[gain_ub,gain_ub,gain_ub,gain_ub,gain_ub,gain_ub,100.,100.,100.,100.,100.])

        linear_constraint = sp.optimize.LinearConstraint(np.ndarray.tolist(linear_constr_arr), [0.],[0.])

        def func(x,d):
            bd = np.matmul(A,x)-d[:,i]
            return np.matmul(bd,bd)

        #def func_jac(x,d):
        #    return A[:,i]

        def fun_hess(x, d):
            return np.zeros(N_unknowns)

        x0 = np.ndarray.tolist(np.ones(N_unknowns))
        d_real=log_mv_real
        d_imag=log_mv_imag
        res_real = sp.optimize.minimize(fun=func, x0=x0,args=(d_real,),method="trust-constr",jac="cs",hess=fun_hess,
                                   constraints=[linear_constraint],tol=1.e-4)
        res_imag = sp.optimize.minimize(fun=func, x0=x0,args=(d_imag,),method="trust-constr",jac="cs",hess=fun_hess,
                                    constraints=[linear_constraint],tol=1.e-4)

        x_rec_real[:,i]=res_real.x
        x_rec_imag[:,i]=res_imag.x

    return measured_vis_array, true_vis_array, gain, x_rec_real, x_rec_imag, x_true_real, x_true_imag



two_by_3_1=Visibilities_grid(2,3,m,ts_2,rg_2,ag_2)
two_by_3_2=Visibilities_grid(2,3,m,ts_2,rg_2_run2,ag_2_run2)
#print (two_by_3[3][:6,0])
#np.save('x_rec_real_automated_constr_0_run2',two_by_3_2[3])
x_rec_real_1=np.load('x_rec_real_automated_constr_0.npy')
x_rec_real_2=np.load('x_rec_real_automated_constr_0_run2',two_by_3_2[3])
x_true_real=two_by_3[5]




'''
x_lstsq=np.linalg.lstsq(A_arr,log_mv_real,rcond=None)[0]

x_lstsq_gain_norm=np.zeros((6,1024))
for i in range(0,1024):
    if np.mean(x_lstsq[:6,i],axis=0)>0.:
        x_lstsq_gain_norm[:,i]=x_lstsq[:6,i]/np.abs(np.mean(x_lstsq[:6,i],axis=0))-1.0
    else:
        x_lstsq_gain_norm[:,i]=x_lstsq[:6,i]/np.abs(np.mean(x_lstsq[:6,i],axis=0))+1.0

print (x_lstsq_gain_norm.shape)
'''

#check the sum of gains for individual time channels

#gain_real_sum_arr=np.array([])
#for i in range(1024):
 #   sum=np.sum(gain_amp[:,i])
  #  gain_real_sum_arr=np.append(gain_real_sum_arr,sum)
#print (gain_real_sum_arr.max())



#unique_bls_real=unique_bls_real+np.abs(unique_bls_real.min())
#unique_bls_real= unique_bls_real.clip(min=0)

#gain_amp=np.log(np.abs(unique_bls_real.min()))*gain_amp
'''
ind=0
print (x_rec_real[:6,ind])
print (x_true[:6,ind],'real_gain')

print (x_lstsq[:6,ind])
print (x_lstsq[:6,ind]/np.abs(np.mean(x_lstsq[:6,ind],axis=0)))
print (np.mean(x_lstsq[:6,ind],axis=0),'mean')
print (x_lstsq_gain_norm[:6,ind],'norm_gain')
x_lstsq_uniq=1.0-x_lstsq[:6,ind]/np.abs(np.mean(x_lstsq[:6,ind],axis=0))
print (np.mean(x_lstsq_gain_norm,axis=0)[0])
'''
plt.plot(x_true_real[:6,:].T,'r')
plt.plot(x_rec_real_1[:6,:].T,'og')

plt.plot(x_rec_real_2[:6,:].T,'ob')
blue_patch = mpatches.Patch(color='blue', label='Recovered gains')
red_patch=mpatches.Patch(color='red', label='Simulated gains')

plt.legend(handles=[blue_patch,red_patch])
plt.xlabel('Time channels')
plt.ylabel('Gains')
plt.show()

fig, ax = plt.subplots()


#ax.scatter(np.mean(x_true[:6,0:500],axis=1),np.mean(x_rec_mat[:6,0:500],axis=1))
#ax.scatter(np.mean(x_true[:6,:],axis=1),np.mean(x_lstsq_gain_norm[:6,:],axis=1))
ax.scatter(np.mean(x_true_real[:6,:],axis=1),np.mean(x_rec_real_1[:6,:],axis=1))
ax.scatter(np.mean(x_true_real[:6,:],axis=1),np.mean(x_rec_real_2[:6,:],axis=1))

plt.show()
