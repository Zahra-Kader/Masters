import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.patches as mpatches
from drift.core import manager
from scipy.sparse import lil_matrix
from random import seed
from random import random

def nan_check(variable):
    shape_before = variable.shape
    print (shape_before[0]*shape_before[1])
    shape_after = variable[np.logical_not(np.isnan(variable))].shape
    return shape_after


def index_find(input_arr,ind_1,ind_2):
    arr_find=np.where(input_arr==[ind_1, ind_2])[0]
    for i in range(len(arr_find)-1):
        if arr_find[i]==arr_find[i+1]:
            location=arr_find[i]
    return location

def prod_ind(ts_file,ind_1,ind_2):
    a_loc=np.where(ts_file['input_a']==ind_1)[0]
    b_loc=np.where(ts_file['input_b']==ind_2)[0]
    for i in a_loc:
        for j in b_loc:
            if i==j:
                location=i
    return location

def vis(ts_file,b):
    return ts_file['vis'][0,b,:]

def log(function):
    zer=function==0.
    function[zer]=1.e-10
    return np.log(function)

def Scatterplot(manager_config_file):
    t=manager_config_file.telescope
    x=t.feedpositions[:,0] #these are x and y positions not x and y polarizations
    y=t.feedpositions[:,1]
    Nfeeds,_=t.feedpositions.shape
    Ndish=Nfeeds/2
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.ylabel('Distance (m)')
    plt.xlabel('Distance (m)')
    for i in range(0,Ndish):
        j=i+Ndish
        ax.annotate((i),(x[i],y[i]))
    plt.show()

def Bls_counts(manager_config_file):
    t=manager_config_file.telescope
    Nfeeds,_=t.feedpositions.shape
    Ndish=Nfeeds/2
    Nbls=Ndish*(Ndish-1)/2
    x=t.feedpositions[:,0] #these are x and y positions not x and y polarizations
    y=t.feedpositions[:,1]

    unique=np.unique(t.baselines,axis=0)
    #N_unique_bls=len(unique) #can't use this if you set auto_correlations=Yes in config file
    a=unique[:,0]
    b=unique[:,1]

    correlation_arr=np.array([])
    for k in range(0,len(b)):
        for i in range(0,Ndish):
            for j in range(0,Ndish):
                if y[j]-y[i]==b[k]:
                    if x[j]-x[i]==a[k]:
                        if i!=j:
                            arr_sing=i,j,a[k],b[k]
                            correlation_arr=np.append(correlation_arr,arr_sing)
    correlation_arr=np.reshape(correlation_arr,(-1,4))
    corr_unique,corr_counts=np.unique(correlation_arr[:,2:4],return_counts=True,axis=0)
    sum_counts=np.cumsum(corr_counts)
    return correlation_arr, sum_counts,corr_counts

def Red_bls(manager_config_file,file_no_gain):
    t=manager_config_file.telescope
    Nfeeds,_=t.feedpositions.shape
    Ndish=Nfeeds/2
    prods=file_no_gain['index_map']['prod'][:]
    x=t.feedpositions[:,0] #these are x and y positions not x and y polarizations
    y=t.feedpositions[:,1]
    unique=np.unique(t.baselines,axis=0)
    #N_unique_bls=len(unique) #can't use this if you set auto_correlations=Yes in config file
    a=unique[:,0]
    b=unique[:,1]

    correlation_indices=np.array([])
    for k in range(0,len(b)):
        for i in range(0,Ndish):
            for j in range(0,Ndish):
                if y[j]-y[i]==b[k]:
                    if x[j]-x[i]==a[k]:
                        if i!=j:
                            if i<j:
                                correlation_indices=np.append(correlation_indices,prod_ind(prods,i,j))
                            else:
                                correlation_indices=np.append(correlation_indices,prod_ind(prods,j,i))
    sum_counts=Bls_counts(manager_config_file)[1]
    red_bls=np.split(correlation_indices,sum_counts)
    return red_bls



def A_matrix(manager_config_file):
    t=manager_config_file.telescope
    x=t.feedpositions[:,0] #these are x and y positions not x and y polarizations
    y=t.feedpositions[:,1]
    Nfeeds,_=t.feedpositions.shape
    Ndish=Nfeeds/2
    correlation_arr=Bls_counts(manager_config_file)[0]
    Nbls,_=correlation_arr.shape
    sum_counts=Bls_counts(manager_config_file)[1]

    N_unique_bls=len(sum_counts)
    N_unknowns=Ndish+N_unique_bls
    A=np.zeros((Nbls,N_unknowns))
    A_phase=np.zeros((Nbls,N_unknowns))
    for n in range(Nbls):
        corr_single_i=np.int(correlation_arr[n,0:2][0])
        corr_single_j=np.int(correlation_arr[n,0:2][1])
        A[n,corr_single_i]=1
        A[n,corr_single_j]=1
        A_phase[n,corr_single_i]=1
        A_phase[n,corr_single_j]=-1


    sum_counts_new=np.append(np.array([0]),sum_counts)
    for i in range(len(sum_counts_new)-1):
        A[sum_counts_new[i]:sum_counts_new[i+1],Ndish+i]=1
        A_phase[sum_counts_new[i]:sum_counts_new[i+1],Ndish+i]=1
    constr_sum=np.append(np.ones(Ndish),np.zeros(N_unique_bls))
    constr_x_orient=np.append(x[:Ndish],np.zeros(N_unique_bls))
    constr_y_orient=np.append(y[:Ndish],np.zeros(N_unique_bls))
    A_no_constr=A
    A_no_constr_phase=A_phase
    A=np.vstack((A,constr_sum))
    A_phase=np.vstack((A_phase,constr_sum, constr_x_orient, constr_y_orient))
    temp = np.linalg.pinv(np.matmul(A_no_constr.T,A_no_constr))
    temp_phase = np.linalg.pinv(np.matmul(A_no_constr_phase.T,A_no_constr_phase))
    error_ID_noise_gain=np.sqrt(np.diag(temp))[:Ndish]
    error_ID_noise_phase=np.sqrt(np.diag(temp_phase))[:Ndish]
    #error_ID_noise_gain_2d=error_2d_ordered(Ndish,manager_config_file,error_ID_noise_gain)

    return A_no_constr, A, A_no_constr_phase, A_phase, temp, error_ID_noise_phase, error_ID_noise_gain

def Noise_cov_matrix(manager_config_file,measured_vis,time_channel,sigma):
    t=manager_config_file.telescope
    Nfeeds,_=t.feedpositions.shape
    Ndish=Nfeeds/2
    correlation_arr=Bls_counts(manager_config_file)[0]
    Nbls,_=correlation_arr.shape
    N_cov=np.array([])
    for i in measured_vis[:,time_channel]:
        N_cov=np.append(N_cov,sigma**2/(np.abs(i))**2)

    N=np.diag(N_cov)
    N_no_constr=np.diag(N_cov)
    #N=np.ones(Nbls)
    #N=np.diag(N)
    #print (N.shape)
    N_amp=np.vstack((N,np.zeros(Nbls)))
    N_phase=np.vstack((N,np.zeros((3,Nbls))))
    #print (N.shape)
    zeros=np.zeros((Nbls+1,1))
    zeros_phase=np.zeros((Nbls+3,3))

    N_amp=np.hstack((N_amp,zeros))
    N_phase=np.hstack((N_phase,zeros_phase))
    N_amp[-1][-1]=1.
    N_phase[-1][-1]=N_phase[-2][-2]=N_phase[-3][-3]=1.
    return N_no_constr,N_amp, N_phase

def lstsq(A_err,A_rec,N_err,N_rec,mv):
    error=np.matmul(A_err.T,np.linalg.pinv(N_err))
    error=np.sqrt(np.diag(np.linalg.pinv(np.matmul(error,A_err))))
    #error_gain_2d=error_2d_ordered(Ndish,manager_config_file,error)

    #print (np.diag(errorbar_small),'square of errorbar')

    x_lstsq_t1=np.matmul(A_rec.T,np.linalg.pinv(N_rec))
    x_lstsq_t1=np.linalg.pinv(np.matmul(x_lstsq_t1,A_rec))
    x_lstsq_t2=np.matmul(A_rec.T,np.linalg.pinv(N_rec))
    #x_lstsq_t1_noiseless=np.linalg.pinv(np.matmul(A.T,A))
    #x_lstsq_t2_noiseless=np.matmul(A.T,log_mv_real_no_noise)
    #x_rec_real=np.matmul(x_lstsq_t1_noiseless,x_lstsq_t2_noiseless)
    #x_lstsq_t2=np.matmul(x_lstsq_t2,log_mv_real)
    x_lstsq_t2=np.matmul(x_lstsq_t2,mv)
    x_rec_real=np.matmul(x_lstsq_t1,x_lstsq_t2)
    return error, x_rec_real
'''
def error_2d_ordered(Ndish,manager_config_file,error):
    t=manager_config_file.telescope
    pos=t.feedpositions[:Ndish,:]
    pos_list=pos.tolist()
    pos_tup = [tuple(row) for row in pos_list]
    dtype=[('x',float),('y',float)]
    arr=np.array(pos_tup,dtype)
    ordered_pos=np.sort(arr,order=['y','x'])
    indices=np.array([])
    for i in range(len(ordered_pos)):
        indices=np.append(indices,index_find(pos,np.int(ordered_pos['x'][i]),np.int(ordered_pos['y'][i])))
    indices=[int(i) for i in indices]
    error_2d=[error[i] for i in indices]
    error_2d=np.flip(np.array(error_2d).reshape(v,u),axis=0)
    return error_2d
'''

def Visibilities_grid(manager_config_file,file_no_gain,file_gain,file_with_gain,file_with_noise,time_channel):
    t=manager_config_file.telescope
    x=t.feedpositions[:,0] #these are x and y positions not x and y polarizations
    y=t.feedpositions[:,1]
    Nfeeds,_=t.feedpositions.shape

    Ndish=Nfeeds/2
    correlation_arr=Bls_counts(manager_config_file)[0]
    Nbls,_=correlation_arr.shape
    rg=file_gain['gain'][0,:,:]
    #print (rg.min(),'rg')
    prods=file_no_gain['index_map']['prod'][:]
    _,Ntimes=file_no_gain['vis'][0,:,:].shape

    dt = file_no_gain['index_map']['time']['ctime'][1]-file_no_gain['index_map']['time']['ctime'][0]

    df = file_no_gain['index_map']['freq']['width'] [0] * 1e6
    ndays=1.  #DON'T SET NDAYS=733. TRY NDAYS=1 FOR NOW, MIGHT HAVE TO EVEN REDUCE THIS, DEPENDING ON HOW OFTEN WE CALIBRATE
    #ndays=1000000
    # Calculate the number of samples
    nsamp = int(ndays * dt * df)
    sigma_gn = 50. / np.sqrt(2 * nsamp)
    print (sigma_gn,'sigma_gn')
    sum_counts=Bls_counts(manager_config_file)[1]
    red_bls=Red_bls(manager_config_file,file_no_gain)

    true_vis=np.array([])
    meas_vis=np.array([])
    meas_vis_gnoise=np.array([])
    for n in range(len(sum_counts)):
        #true_vis_array[n,:]=vis(file_no_gain,red_bls[n][0])
        true_vis=np.append(true_vis,vis(file_no_gain,red_bls[n][0]))
        for i in red_bls[n]:
            #measured_vis_array[i,:]=vis(file_with_gain,i)
            meas_vis=np.append(meas_vis,vis(file_with_gain,i))
            meas_vis_gnoise=np.append(meas_vis_gnoise,vis(file_with_noise,i))
            #plt.plot(vis(file_no_gain,i))
        #plt.show()
    meas_vis_no_noise=np.reshape(meas_vis,(Nbls,Ntimes))
    meas_vis_gnoise=np.reshape(meas_vis_gnoise,(Nbls,Ntimes))

    mu, sigma = 0, 1.e2  #6e-6 to get values within 1 sigma  #sigma=Tsys/np.sqrt(delta_nu*t_int) where Tsys=50 K, delta_nu=400 kHz (width of a single channel=400 MHz/1024=0.4 MHz,
                        #t_int=10 s, sigma=0.025,
    print (sigma,'sigma')
    N_real=np.random.normal(mu, sigma, Nbls)
    N_imag=np.random.normal(mu, sigma, Nbls)
    N_comp=np.array([])
    for i in range(len(N_real)):
        N_comp=np.append(N_comp,complex(N_real[i],N_imag[i]))
    #N=sigma*np.random.standard_normal(Nbls)
    #N=sigma*np.ones(Nbls)
    N_col=np.reshape(N_comp,(len(N_comp),1))
    meas_vis=meas_vis_no_noise+N_col
    #print ((np.abs(measured_vis_array_gnoise[:,time_channel])**2).min(),'min gnoise')
    #zer=measured_vis_array==0.
    #measured_vis_array[zer]=1.e-5
    N_unique_bls=len(sum_counts)#len(true_vis_array)/Ntimes
    true_vis=np.reshape(true_vis,(N_unique_bls,Ntimes))
    #print ((np.abs(true_vis[:,time_channel])**2).min(),'min true vis')

    gain=rg[:Ndish,:]
    gain_amp=log(gain).real
    gain_phase=log(gain).imag


    mv_real, mv_real_no_noise, mv_real_gnoise=log(np.abs(meas_vis)), log(np.abs(meas_vis_no_noise)), log(np.abs(meas_vis_gnoise))
    mv_imag, mv_imag_no_noise, mv_imag_gnoise=np.angle(meas_vis), np.angle(meas_vis_no_noise), np.angle(meas_vis_gnoise)

    constr_amp_sum=np.sum(gain_amp[:,time_channel])*np.ones(Ntimes)
    constr_phase_sum=np.sum(gain_phase[:,time_channel])*np.ones(Ntimes)
    constr_phase_x=np.dot(x[:Ndish],gain_phase)*np.ones(Ntimes)
    constr_phase_y=np.dot(y[:Ndish],gain_phase)*np.ones(Ntimes)

    mv_real, mv_real_no_noise, mv_real_gnoise=np.vstack((mv_real,constr_amp_sum)), np.vstack((mv_real_no_noise,constr_amp_sum)), np.vstack((mv_real_gnoise,constr_amp_sum))
    mv_imag, mv_imag_no_noise, mv_imag_gnoise=np.vstack((mv_imag,constr_phase_sum,constr_phase_x,constr_phase_y)), np.vstack((mv_imag_no_noise,constr_phase_sum,constr_phase_x,constr_phase_y)), np.vstack((mv_imag_gnoise,constr_phase_sum,constr_phase_x,constr_phase_y))



    #np.save(N_col,'columned_noise')

    tv_real=log(np.abs(true_vis))
    tv_imag=np.angle(true_vis)
    #print (np.sum(log_gain_amp[:6,time_channel]),'actual sum of gains')
    #A = lil_matrix((Nbls, N_unknowns))
    A_no_constr, A, A_no_constr_phase, A_phase, _,_,_=A_matrix(manager_config_file)
    #N=np.matmul(N_col,N_col.T)
    N_no_constr,N, N_phase=Noise_cov_matrix(manager_config_file,meas_vis,time_channel,sigma)
    N_no_constr_gnoise,N_gnoise, N_gnoise_phase=Noise_cov_matrix(manager_config_file,meas_vis_gnoise,time_channel,sigma_gn)

    error,x_rec_real=lstsq(A_no_constr, A, N_no_constr, N, mv_real)
    error_imag,x_rec_imag=lstsq(A_no_constr_phase, A_phase, N_no_constr, N_phase, mv_imag)

    error_gnoise_imag,x_rec_imag_gnoise=lstsq(A_no_constr_phase, A_phase, N_no_constr_gnoise, N_gnoise_phase, mv_imag_gnoise)
    error_gnoise,x_rec_real_gnoise=lstsq(A_no_constr, A, N_no_constr_gnoise, N_gnoise, mv_real_gnoise)

    #print (np.min(np.abs(measured_vis_array[:,time_channel])**2),'min abs measured vis squared')

    #print (N.shape)
    #print (N)
    #A=A.tocsr()
    x_true_real=np.vstack((gain_amp,tv_real))
    x_true_imag=np.vstack((gain_phase,tv_imag))
    #log_tv_imag=log(np.abs(true_vis_array.imag))
    #N_small=np.ones((len(N_small),len(N_small)))

    #N=sigma*np.identity(Nbls+1)
    #N_small=sigma*np.identity(Nbls)
    #print (np.diag(N_small),'small N')
    x_rec_real_no_noise=np.linalg.lstsq(A,mv_real_no_noise,rcond=None)[0]
    x_rec_imag_no_noise=np.linalg.lstsq(A_phase,mv_imag_no_noise,rcond=None)[0]

    '''Change with trying errorbar as [AtA]^-1 and trying x_rec_real=[AtA]-1 At(log measured vis), i.e. np.linalg.lstsq using measured vis with noise'''
    #errorbar_small=np.linalg.pinv(np.matmul(A_small.T,A_small))
    #errorbar_small=np.sqrt(np.diag(errorbar_small))
    #x_rec_real=np.linalg.lstsq(A,log_mv_real_no_noise,rcond=None)[0]

    #print (np.sum(x_rec_real_lstsq[:6,time_channel]),'sum_gains_recovered')
    mv_real_recovered=np.matmul(A,x_rec_real)
    mv_real_recovered_no_noise=np.matmul(A,x_rec_real_no_noise)
    #errorbar_small=np.sqrt(np.abs(np.diag(x_lstsq_t1)))
    #avg=np.array([])
    #for i in range(Ndish):
        #row_avg=np.mean(err[:25,:25][-(i+1),:])
        #col_avg=np.mean(err[:25,:25][:,i])
        #tot_avg=(row_avg+col_avg)/2
        #line=np.unique(np.append(error[:Ndish,:Ndish][:,i],error[:Ndish,:Ndish][i,:]))
        #line_avg=np.mean(error[:Ndish,:Ndish],axis=0)
        #line_avg=np.mean(line)
        #avg=np.append(avg,line_avg)
    #avg=np.mean(errorbar_small[:Ndish,:Ndish],axis=0)
    #avg=np.flip(avg.reshape(Ndish),axis=0)
    return meas_vis, meas_vis_no_noise, meas_vis_gnoise, true_vis, N_comp, x_rec_real, x_rec_real_no_noise, x_true_real, x_rec_real_gnoise, x_rec_imag, x_rec_imag_no_noise, x_true_imag, x_rec_imag_gnoise, error_gnoise, error_gnoise_imag, error, error_imag, sigma, sigma_gn

def colour_scatterplot(manager_config_file,error_1d):
    t=manager_config_file.telescope
    Nfeeds,_=t.feedpositions.shape
    Ndish=Nfeeds/2
    x=t.feedpositions[:,0][:Ndish] #these are x and y positions not x and y polarizations
    y=t.feedpositions[:,1][:Ndish]
    plt.scatter(x,y,c=error_1d,s=200)
    plt.xlabel('Antenna x-location',fontsize=12); plt.ylabel('Antenna y-location',fontsize=12);plt.title('Relative amplitude error')
    plt.colorbar()
    plt.show()


'''

        #log_mv_imag=log_of_func(np.abs(measured_vis_array.imag))



        #x_true_imag=np.vstack((log_gain_phase,log_tv_imag))

        measured_vis_array=np.array([true[0]*amp[0]*amp[1],true[0]*amp[1]*amp[2],true[0]*amp[3]*amp[4],true[0]*amp[4]*amp[5],
        true[1]*amp[0]*amp[2],true[1]*amp[3]*amp[5],
        true[2]*amp[0]*amp[5],
        true[3]*amp[1]*amp[3],true[3]*amp[2]*amp[4],
        true[4]*amp[0]*amp[3],true[4]*amp[1]*amp[4],true[4]*amp[2]*amp[5],
        true[5]*amp[0]*amp[4],true[5]*amp[1]*amp[5],
        true[6]*amp[0]*amp[5]])
        print (measured_vis_array,'measured vis should be')


        log_mv_real=log_of_func(measured_vis_array)
        lin_constr_amplitude_sum_value=np.sum(log_gain_amp)#*np.ones(Ntimes)
        log_mv_real=np.append(log_mv_real,lin_constr_amplitude_sum_value)
        log_tv_real=log_of_func(true)
        x_true_real=np.append(log_gain_amp,log_tv_real)




        len_mv_diag=len(log_mv_real)

        N=np.ones(len_mv_diag-1)
        N=np.append(N,np.array([2.]))
        N=np.diag(N)

        N=np.array([])
        for _ in range(len_mv_diag):
            value = random()
            N=np.append(N,value)
        N=np.diag(N)


        x_rec_real_lstsq_meth=np.array([])

        for i in range(Ntimes):
            x_norm=x_rec_real_lstsq[:Ndish,i]/np.mean(x_rec_real_lstsq[:Ndish,i],axis=0)
            x_rec_real_lstsq_meth=np.append(x_rec_real_lstsq_meth,1.-x_norm)
        x_rec_real_lstsq=x_rec_real_lstsq_meth.reshape(Ndish,Ntimes)

        def sum(func):
            sum_gain_arr=np.array([])
            for i in range(Ntimes):
                gain=func[:Ndish,i]
                sum_gain_arr=np.append(sum_gain_arr,np.sum(gain))
            return sum_gain_arr

        sum_gamp_arr=sum(x_true_real)
        sum_gphase_arr=sum(x_true_imag)

        def sum_phase_extra_constr(pos):
            sum_gphase_arr=np.array([])
            for i in range(Ntimes):
                for j in range(Ndish):
                    phase=x_true_imag[j,i]*pos[j]
                sum_gphase_arr=np.append(sum_gphase_arr,np.sum(phase))
            return sum_gphase_arr

        sum_x_phase=sum_phase_extra_constr(x)
        sum_y_phase=sum_phase_extra_constr(y)

        rows,cols=log_mv_real.shape
        linear_constr_arr=np.append(np.ones(Ndish),np.zeros(N_unique_bls))

        linear_constraints_x_phase=np.append(x[:Ndish],np.zeros(N_unique_bls))
        linear_constraints_y_phase=np.append(y[:Ndish],np.zeros(N_unique_bls))
        bounds_phase_min=np.ndarray.tolist(np.append(-np.pi*np.ones(Ndish),np.zeros(N_unique_bls)))
        bounds_phase_max=np.ndarray.tolist(np.append(np.pi*np.ones(Ndish),np.zeros(N_unique_bls)))

        for i in range(cols):

            #bounds=sp.optimize.Bounds([gain_lb,gain_lb,gain_lb,gain_lb,gain_lb,gain_lb,-100.,-100.,-100.,-100.,-100.],
                                      #[gain_ub,gain_ub,gain_ub,gain_ub,gain_ub,gain_ub,100.,100.,100.,100.,100.])
            bounds_phase=sp.optimize.Bounds(bounds_phase_min,bounds_phase_max)
            linear_constraint_amp = sp.optimize.LinearConstraint(np.ndarray.tolist(linear_constr_arr), [sum_gamp_arr[i]],[sum_gamp_arr[i]])
            linear_constraint_phase = sp.optimize.LinearConstraint(np.ndarray.tolist(linear_constr_arr), [sum_gphase_arr[i]],[sum_gphase_arr[i]])
            linear_constraints_x_phase_arr=sp.optimize.LinearConstraint(np.ndarray.tolist(linear_constraints_x_phase), [sum_x_phase[i]],[sum_x_phase[i]])
            linear_constraints_y_phase_arr=sp.optimize.LinearConstraint(np.ndarray.tolist(linear_constraints_y_phase), [sum_y_phase[i]],[sum_y_phase[i]])

            def func(x,d):
                bd = np.matmul(A,x)-d[:,i]
                return np.matmul(bd,bd)

            #def func_jac(x,d):
            #    return A[:,i]

            def fun_hess(x, d):
                return np.zeros(N_unknowns)

            x0 = np.ndarray.tolist(np.ones(N_unknowns))

            res_real = sp.optimize.minimize(fun=func, x0=x0,args=(log_mv_real,),method="trust-constr",jac="cs",hess=fun_hess,
                                       constraints=[linear_constraint_amp],tol=1.e-4)
            res_imag = sp.optimize.minimize(fun=func, x0=x0,args=(log_mv_imag,),method="trust-constr",jac="cs",hess=fun_hess, bounds=bounds_phase,
                                        constraints=[linear_constraint_phase],tol=1.e-4)

            x_rec_real[:,i]=res_real.x
            x_rec_imag[:,i]=res_imag.x

        log_mv_real_rec=np.matmul(A,x_rec_real)
        log_mv_real_imag=np.matmul(A,x_rec_imag)
'''
