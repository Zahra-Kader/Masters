#see bottom for plotting analytical function nicely
import numpy as np
import matplotlib.pyplot as plt


#to plot output from form_factor_analytical_rewritten.py
spec='tt'
expt='planck'
lmax=16384
delta_l=4

delta_theta=np.pi/lmax


if spec=='eb' or spec=='tb':
    estimators=['shear_plus', 'shear_cross']
    estimators_full=['shear plus', 'shear cross']
    norm=1/(delta_theta**2)
else:
    estimators=['conv', 'shear_plus', 'shear_cross']
    estimators_full=['convergence','shear plus', 'shear cross']
    norm=2/(delta_theta**2)


working_dir_aa='/Users/Heather/Documents/HeatherVersionPlots/FormFactor/IntegralByConvolution'
working_dir_orig='/Users/Heather/Documents/HeatherVersionPlots/planck/FormFactor/'


"""
def plot_ff_straight_integral(lmax_plot):
    for i,  est in enumerate(estimators):
        Lff=np.loadtxt(working_dir_orig+'/Datafiles/ff_'+est+'_'+spec)
        L=Lff[:,0]
        ff=Lff[:,1]
        plt.plot(L, ff, '-o', label=est)
    
    
    plt.legend()
    plt.xlim(0, lmax_plot)
    plt.ylim(-0.2,1.2)
    plt.title('Multiplicative bias for '+spec.upper()+' real space estimators')
    plt.savefig(working_dir_orig+'/Plots/ff_'+spec+'_'+expt+'_lmax'+str(lmax_plot))
    plt.show()


plot_ff_straight_integral(1000)
"""

LF_H=np.loadtxt('/Users/Heather/Documents/HeatherVersionPlots/planck/FormFactor/DataFiles/ff_conv_tt')
L_H=LF_H[:,0]
F_H=LF_H[:,1]
LF_H1=np.loadtxt('/Users/Heather/Documents/HeatherVersionPlots/planck/FormFactor/DataFiles/ff_shear_plus_tt')
L_H1=LF_H1[:,0]
F_H1=LF_H1[:,1]


Lff_noise=np.loadtxt(working_dir_aa+'/Datafiles/ff_tt_conv_alternative_analytical_'+expt+'_lmax16384_deltal4_with_noise')
L_noise=Lff_noise[:,0]
ff_noise=Lff_noise[:,1]
plt.plot(L_noise, ff_noise*norm, label='conv with noise', color='y')
colours=['b','r','g']
def plot_ff_alternative_analytical(lmax_plot):
    for i,  est in enumerate(estimators):
        if est=='conv' or est=='shear_plus':
            Lff=np.loadtxt(working_dir_aa+'/Datafiles/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l))
        else:
            Lff=np.loadtxt(working_dir_aa+'/Datafiles/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l)+'_other_axis')
        L=Lff[:,0]
        ff=Lff[:,1]
        plt.plot(L, ff*norm, label=estimators_full[i], color=colours[i])
        """
            if spec=='te' or spec=='ee':
            Lff_other_axis=np.loadtxt(working_dir_aa+'/Datafiles/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax4096_deltal1_other_axis')
            L_oa=Lff_other_axis[:,0]
            ff_oa=Lff_other_axis[:,1]
            plt.plot(L_oa, ff_oa/ff_oa[1], '--', label=est+' other axis', color=colours[i])
            """
        
        Lff_full_integral=np.loadtxt(working_dir_orig+'/Datafiles/ff_'+est+'_'+spec)
        L_fi=Lff_full_integral[:,0]
        ff_fi=Lff_full_integral[:,1]
        #plt.plot(L_fi, ff_fi, 'o', label=est+' full integral', color=colours[i])
        
        """if est=='shear_plus' and spec=='tt':
            norm=1/ff_maps_shear_RS[1]#*((ff/ff[1])[450])
            plt.errorbar(L_maps_shear,ff_maps_shear_RS*norm,sigma_maps_shear_RS*norm,fmt='o',linestyle='None', label='from map code', color='y')
            print ff_maps_shear_RS*norm
            print norm"""


    plt.legend()
    plt.xlim(0, lmax_plot)
    plt.ylim(-0.4,1.2)
    if expt=='ref' or expt=='act' and lmax_plot<2000 and spec=='tt':
        plt.ylim(0,3)
    elif expt=='ref' or expt=='act' and lmax_plot>=2000 and spec=='tt':
        plt.ylim(0,30)
    elif expt=='ref_foregrounds' and lmax_plot<2000 and spec=='tt':
        plt.ylim(0,3)
    elif expt=='ref_foregrounds' and lmax_plot>=2000 and spec=='tt':
        plt.ylim(0,12)
    elif spec=='eb':
        plt.ylim(0,1.2)
    elif spec=='ee':
        plt.ylim(-0.2,1.2)
    plt.title('Multiplicative Bias for '+spec.upper()+' Real Space Estimators')
    plt.savefig(working_dir_aa+'/FinalPlots/ff_'+spec+'_'+expt+'_lmax'+str(lmax_plot)+'_just_curve')

    plt.show()

plot_ff_alternative_analytical(1000)
plot_ff_alternative_analytical(4000)


#plot form factors for varying beam size
theta_fwhm=[7.,10.]
est='conv'
Lff=np.loadtxt(working_dir_aa+'/Datafiles/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l))
L=Lff[:,0]
ff=Lff[:,1]
plt.plot(L, ff*norm, label='no beam')
for t in theta_fwhm:
    Lff=np.loadtxt(working_dir_aa+'/Datafiles/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l)+'_with_noise_thetafwhm'+str(int(t)))
    L=Lff[:,0]
    ff=Lff[:,1]
    plt.plot(L, ff*norm, label='theta_fwhm= '+str(int(t))+' arcminutes')
plt.title('Multiplicative Bias for '+spec.upper()+'Convergence Real Space Estimators')
plt.legend()
plt.show()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import sys
sys.exit()


#Plots output from map code - not best to use coz depends heavlily on field of view and resolution
Lff_H_a_new=np.loadtxt('/Users/Heather/Documents/HeatherVersionPlots/FormFactor/IntegralByConvolution/Datafiles/ff_tt_conv_alternative_analytical_planck')
L_H_a_new=Lff_H_a_new[:,0]
ff_H_a_new=Lff_H_a_new[:,1]

Lff_H_a=np.loadtxt('/Users/Heather/Documents/HeatherVersionPlots/planck/FormFactor/tt_conv_data_file')
L_H_a=Lff_H_a[:,0]
ff_H_a=Lff_H_a[:,1]

Lff_H_a1=np.loadtxt('/Users/Heather/Documents/HeatherVersionPlots/planck/FormFactor/tt_conv_datafile20feb15')
L_H_a1=Lff_H_a1[:,0]
ff_H_a1=Lff_H_a1[:,1]

Lff_k_c=np.loadtxt('/Users/Heather/Documents/HeatherVersionPlots/planck/FormFactor/ff_kavi_lmax4500.txt')
L_k_c=Lff_k_c[:,0]
ff_k_c=Lff_k_c[:,1]

Lff_jet=np.loadtxt('/Users/Heather/Documents/HeatherVersionPlots/planck/fov20/FormFactor/tt_conv_data_file_jet_maps')
L_jet=Lff_jet[:,0]
ff_jet=Lff_jet[:,1]

Lff_m_map=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/tt_conv_datafile_feb15')#tt_conv_data_file')
L_m_map=Lff_m_map[:,0]
ff_m_map=Lff_m_map[:,1]

Lff_m_map1=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/ff_tt_conv_datafile20feb15')#tt_conv_data_file')
L_m_map1=Lff_m_map1[:,0]
ff_m_map1=Lff_m_map1[:,1]
sd_m_map1=Lff_m_map1[:,2]

Lff_m_map2=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/ff_tt_conv_datafile_fov60_nside2048_n15')#tt_conv_data_file')
L_m_map2=Lff_m_map2[:,0]
ff_m_map2=Lff_m_map2[:,1]
sd_m_map2=Lff_m_map2[:,2]

Lff_m_map3=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/ff_tt_conv_datafile_fov40_nside2048_n15and3mix')#tt_conv_data_file')
L_m_map3=Lff_m_map3[:,0]
ff_m_map3=Lff_m_map3[:,1]
sd_m_map3=Lff_m_map3[:,2]

Lff_m_map4=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_heather_planck_tt_conv_20_2048_datafile_amp10overk')#ff_tt_conv_datafile_fov80_nside2048_n15')#tt_conv_data_file')
L_m_map4=Lff_m_map4[:,0]
ff_m_map4=Lff_m_map4[:,1]
sd_m_map4=Lff_m_map4[:,2]

Lff_m_map6=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_heather_planck_tt_conv_20_2048_datafile_amp1overk')#ff_tt_conv_datafile_fov80_nside2048_n15')#tt_conv_data_file')
L_m_map6=Lff_m_map6[:,0]
ff_m_map6=Lff_m_map6[:,1]
sd_m_map6=Lff_m_map6[:,2]

Lff_m_map5=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/ff_tt_conv_datafile_fov40_nside2048_n3_amp1overk')
L_m_map5=Lff_m_map5[:,0]
ff_m_map5=Lff_m_map5[:,1]
sd_m_map5=Lff_m_map5[:,2]

Lff_spec=np.loadtxt('/Users/Heather/Documents/HeatherVersionPlots/planck/fov20/FormFactor/tt_conv_planck_form_factor_from_spectra_data_file')
L_spec=Lff_spec[:,0]
ff_spec=Lff_spec[:,1]

Lff_H_map=np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/HeatherVersion/FormFacMaps/FFdatafiles/ff_planck_tt_conv_40_1024')
L_H_map=Lff_H_map[:,0]
ff_H_map=Lff_H_map[:,1]
sd_H_map=Lff_H_map[:,2]

Lff_H_map1=np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/HeatherVersion/FormFacMaps/FFdatafiles/ff_planck_tt_conv_20_1024')
L_H_map1=Lff_H_map1[:,0]
ff_H_map1=Lff_H_map1[:,1]
sd_H_map1=Lff_H_map1[:,2]

Lff_H_map2=np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/HeatherVersion/FormFacMaps/FFdatafiles/ff_planck_tt_conv_20_1024_4096_amp1')
L_H_map2=Lff_H_map2[:,0]
ff_H_map2=Lff_H_map2[:,1]
sd_H_map2=Lff_H_map2[:,2]

Lff_H_map3=np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/HeatherVersion/FormFacMaps/FFdatafiles/ff_planck_tt_conv_40_2048_4096_goliath')
L_H_map3=Lff_H_map3[:,0]
ff_H_map3=Lff_H_map3[:,1]
sd_H_map3=Lff_H_map3[:,2]

Lff_H_map4=np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/HeatherVersion/FormFacMaps/FFdatafiles/ff_planck_tt_conv_40_2048_4096_goliath_amp1')
L_H_map4=Lff_H_map4[:,0]
ff_H_map4=Lff_H_map4[:,1]
sd_H_map4=Lff_H_map4[:,2]


delta_L=float(L_H_a_new[1]-L_H_a_new[0])
index_array=np.concatenate((np.array([1]), (np.array([18.,   90.,  180.,  270.,  360.,  450.,  540.,  630.,  720.,  810.,  900., 990.])/delta_L))).astype('int')
#index_array=np.concatenate((np.array([1]), (np.array([18.,   90.,  270.,  450.,  630.,  810.,   990.])/delta_L))).astype('int')

A10=1/(10*(40*np.pi/(180*2048))**1)
A1=1/(1*(40*np.pi/(180*2048))**1)
A10_20=1/(10*(20*np.pi/(180*2048))**1)
A1_20=1/(1*(20*np.pi/(180*2048))**1)
A10_60=1/(10*(60*np.pi/(180*2048))**1)
ff_m_map1_norm=ff_m_map1/ff_m_map1[11]*ff_H_a[4]
normfac2=ff_m_map1_norm[7]+(ff_m_map1_norm[8]-ff_m_map1_norm[7])*(L_m_map2[5]-L_m_map1[7])/(L_m_map1[8]-L_m_map1[7])

plt.plot(L_H_a_new,ff_H_a_new/ff_H_a_new[1], label='analytical from py new method')

plt.plot(L_m_map1, ff_m_map1*A10, 'o',label='map fov=40 nside=1024')# n=15')

#plt.plot(L_H_map1, ff_H_map1/ff_H_map1[4]*ff_H_a[4], 'o', label='map fov=20 nside=1024 nside_cmb=4096, amp=10/k')
#plt.plot(L_H_map2, ff_H_map2/ff_H_map2[6]*ff_H_a[4], 'o', label='map fov=20 nside=1024 nside_cmb=4096, amp=1/k')
#plt.plot(L_H_map3, ff_H_map3/ff_H_map3[3]*ff_H_a[3], 'o', label='map fov=40 nside=2048 nside_cmb=4096, amp=10/k')
#plt.plot(L_H_map4, ff_H_map4/ff_H_map4[3]*ff_H_a[3], 'o', label='map fov=40 nside=2048 nside_cmb=4096, amp=1/k')

#plt.plot(L_H_map, ff_H_map/ff_H_map[3]*ff_H_a[4], 'o', label='map fov=40 nside=1024 nside_cmb=1024')

plt.plot(L_m_map3, ff_m_map3*A10, 'o', label='map fov=40 nside=2048 amp=10/k')# n=15')#/ff_m_map3[8]*ff_H_a[4]
plt.plot(L_m_map5, ff_m_map5*A1, 'o', label='map fov=40 nside=2048 amp=1/k')#/ff_m_map5[3]*ff_H_a[4]
#plt.plot(L_m_map2, ff_m_map2*A10_60, 'o', label='map fov=60 nside=2048')# n=15')
plt.plot(L_m_map4, ff_m_map4*A10_20, 'o', label='map fov=20 nside=2048 amp=10/k')# n=15')#/ff_m_map4[5]*ff_H_a[3]
plt.plot(L_m_map6, ff_m_map6*A1_20, 'o', label='map fov=20 nside=2048 amp=1/k')# n=15')#/ff_m_map6[4]*ff_H_a[3]

#plt.errorbar(L_m_map3, ff_m_map3/ff_m_map3[10]*ff_H_a[4], sd_m_map3/ff_m_map3[7]*ff_H_a[4],fmt='o',linestyle='None', label='map fov=40 nside=2048')

#plt.plot(L_H_a_new[index_array],ff_H_a_new[index_array]/ff_H_a_new[1], label='analytical from py new method')
#plt.plot(L_k_c, ff_k_c/ff_k_c[0], label='analytical from kavi.c')
#plt.plot(L_H_a,ff_H_a, 'o',label='analytical from py')
#plt.plot(L_H_a1,ff_H_a1,'o', label='analytical from py')
#plt.plot(L_jet, ff_jet, label='Jet from maps')
#plt.plot(L_m_map[1:,], ff_m_map[1:,]/ff_m_map[8]*ff_H_a[5], label='Martin from maps previous run of code')
#plt.plot(L_m_map, ff_m_map/ff_m_map[3]*ff_H_a[4], label='Martin from maps')
#plt.plot(L_spec, np.sqrt(ff_spec), label='sqrt ratio of input and output spectra')
plt.xlim(0,1000)
plt.ylim(0, 1.2)
#plt.xscale('log')
#plt.yscale('log')

plt.legend()
plt.title('Form Factor tt convergence')
plt.show()
plt.close()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -




#Plots output from my analytic code using the full integral. This should match the alternative analytic plot
spec='tt'
est='conv'
expt='act'
k_list=np.array([1,5,10, 30, 50,70, 90, 110, 130,150,170,190,210,230,250,270,290,310, 330,350])#add 20 for planck if you want#k_list=np.array([0.5,1,2,5,7,10,15, 20,30,40,50, 70, 90]) #can't use k<1 or don't have periodic boundary conditions
n_iter=60

#f_norm=np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/form_factor_package_Martin/Heather/form_factor_results_'+spec+'_k50')-np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/form_factor_package_Martin/Heather/form_factor_results_null_'+spec+'_k50')
#norm=ff_H_a[4]/f_norm.mean()

#plt.plot(L_H_a_new,ff_H_a_new/ff_H_a_new[1], label='analytical from py new method')

#plt.plot(L_H_a,ff_H_a/ff_H_a[1], 'o', label='analytical')
#plt.plot(450, f_norm.mean()*norm, 'D')
#plt.plot(L_m_map3, ff_m_map3/ff_m_map3[8]*ff_H_a[4], 'D', label='map fov=40 nside=2048 amp=10/k', color='r')

#plt.plot(450, ff_H_a[4], 'o', color='g', label='map fov=40 nside=2048 amp=0.1/k')

to_norm=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_'+spec+'_k50_amp1_HS_lots')-np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_null_'+spec+'_k50_amp1_HS_lots')
if expt=='act':
    Lff_H_a_new=np.loadtxt('/Users/Heather/Documents/HeatherVersionPlots/FormFactor/IntegralByConvolution/Datafiles/ff_tt_conv_alternative_analytical_ref_lmax4096_deltal1')
else:
    Lff_H_a_new=np.loadtxt('/Users/Heather/Documents/HeatherVersionPlots/FormFactor/IntegralByConvolution/Datafiles/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax4096_deltal1')
L_H_a_new=Lff_H_a_new[:,0]
ff_H_a_new=Lff_H_a_new[:,1]
plt.plot(L_H_a_new,ff_H_a_new/ff_H_a_new[1], label='Real Space Estimator Analytic')
norm=1/np.mean(to_norm)
Npix=1
A=Npix*40*np.pi/(180*2048)
print 'Norm l=450:',norm
print 'A inverse', 1/A
for k in k_list:
    theta_max=(np.pi/180.)*40.
    renorm=2.*np.pi/theta_max
    L=renorm*k
    if k<100 and expt=='planck':
        f=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_'+spec+'_k'+str(int(k))+'_amp1_lots')
        f_null=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_null_'+spec+'_k'+str(int(k))+'_amp1_lots')
    elif k<180 and expt=='planck':
        f=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_lots')
        f_null=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_null_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_lots')
    else:
        f=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_'+expt+'_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_lots')
        f_null=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_null_'+expt+'_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_lots')
    
    #print 'f:', f,'; f_null:',f_null
    f=f[0:n_iter]*norm#/(-1*A)
    f_null=f_null[0:n_iter]*norm#/(-1*A)
    print 'mean f', np.mean(f)
    print 'mean f null', np.mean(f_null)
    """
    if k==1:
        plt.plot(L, np.mean(f), 'D', color='y', label='Real Space Estimator from lensed map')
        plt.plot(L, np.mean(f_null), 'D', color='r', label='Real Space Estimator from unlensed map')
        #plt.plot([L]*n_iter, (f), 'o', color='y', label='Real Space Estimator from lensed map')
        #plt.plot([L]*n_iter, (f_null), 'o', color='r', label='Real Space Estimator from unlensed map')
    else:
        plt.plot(L, np.mean(f), 'D', color='y')
        plt.plot(L, np.mean(f_null), 'D',color='r')
        #plt.plot([L]*n_iter, (f), 'o', color='y')
        #plt.plot([L]*n_iter, (f_null), 'o', color='r')
    """
    f-=f_null
    print 'Real space k=',k,'average=',np.mean(f/A)


    if k<100 and expt=='planck':
        f_HS=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_'+spec+'_k'+str(int(k))+'_amp1_HS_lots')
        f_null_HS=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_null_'+spec+'_k'+str(int(k))+'_amp1_HS_lots')
    elif k<180 and expt=='planck':
        f_HS=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_HS_lots')
        f_null_HS=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_null_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_HS_lots')
    else:
        f_HS=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_'+expt+'_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_HS_lots')
        f_null_HS=np.loadtxt('/Users/Heather/Documents/Code/form_factor_package_Martin/Heather/form_factor_results_null_'+expt+'_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_HS_lots')
    f_HS-=f_null_HS
    print 'Harmonic space k=',k,'average=',np.mean(f_HS)
    f_HS*=norm#/=A*(-1)
    f_HS=f_HS[0:n_iter]
    #print 'f:',np.mean(f),'; f_HS:',np.mean(f_HS)


    
    """
    if k==1:
        plt.plot([L]*n_iter, (f), 'o', color='g', label='Real Space Estimator from maps')
        #plt.plot([L]*n_iter, (f_HS), 'o', color='b', label='Harmonic space estimator from maps')
    else:
        plt.plot([L]*n_iter, (f), 'o', color='g')
        #plt.plot([L]*n_iter, (f_HS), 'o', color='b')
    """
    if k==1:
        plt.errorbar(L, np.mean(f), np.sqrt(f.var()), fmt='o', linestyle='None', color='b', label='Real Space Estimator Maps')#label='Real Space maps 60 iterations, fov=40, amp=1/k')
        plt.errorbar(L, np.mean(f_HS), np.sqrt(f_HS.var()), fmt='o', linestyle='None', color='g', label='Harmoic Space Estimator Maps')#label='Harmonic space maps 60 iterations, fov=40, amp=1/k')
    else:
        plt.errorbar(L, np.mean(f), np.sqrt(f.var()), fmt='o', linestyle='None', color='b')
        plt.errorbar(L, np.mean(f_HS), np.sqrt(f_HS.var()), fmt='o', linestyle='None', color='g')#, label='Harmonic space estimator from maps')




#to compare to earlier results:
"""
k_list=np.array([1,2,5,7,10,15, 20,30,40,50, 70, 90])
n_iter=15
to_norm=np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/form_factor_package_Martin/Heather/form_factor_results_'+spec+'_k50_amp1_HS')-np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/form_factor_package_Martin/Heather/form_factor_results_null_'+spec+'_k50_amp1_HS')

norm=1/np.mean(to_norm)
norm=1/(0.2*(20*np.pi/(180*1024))**1)
for k in k_list:
    f=np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/form_factor_package_Martin/Heather/form_factor_results_'+spec+'_k'+str(int(k))+'_amp1')
    f_null=np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/form_factor_package_Martin/Heather/form_factor_results_null_'+spec+'_k'+str(int(k))+'_amp1')
    print 'f:', f,'; f_null:',f_null
    f-=f_null
    f*=norm*(-1)
    f=f[0:n_iter]    #0-5 is amp 0.1, 6-11 is amplitude 1
    
    f_HS=np.real(np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/form_factor_package_Martin/Heather/form_factor_results_'+spec+'_k'+str(int(k))+'_amp1_HS'))
    f_null_HS=np.real(np.loadtxt('/Users/Heather/Dropbox/PolRealSp/Codes/form_factor_package_Martin/Heather/form_factor_results_null_'+spec+'_k'+str(int(k))+'_amp1_HS'))
    f_HS-=f_null_HS
    f_HS*=norm*(-1)
    f_HS=f_HS[0:n_iter]
    #print 'f:',np.mean(f),'; f_HS:',np.mean(f_HS)
    
    theta_max=(np.pi/180.)*20.
    renorm=2.*np.pi/theta_max
    L=renorm*k
    
    """"""
        if k==1:
        plt.plot([L]*n_iter, (f), 'o', color='g', label='Real Space Estimator from maps')
        plt.plot([L]*n_iter, (f_HS), 'o', color='b', label='Harmonic space estimator from maps')
        else:
        plt.plot([L]*n_iter, (f), 'o', color='g')
        plt.plot([L]*n_iter, (f_HS), 'o', color='b')
        """"""
    if k==1:
        plt.errorbar(L, np.mean(f), np.sqrt(f.var()), fmt='o', linestyle='None', color='r', label='Real Space maps 15 iterations, fov=20, amp=0.2/k, n=1024')
        plt.errorbar(L, np.mean(f_HS), np.sqrt(f_HS.var()), fmt='o', linestyle='None', color='y', label='Harmonic space maps 15 iterations, fov=20, amp=0.2/k, n=1024')
    else:
        plt.errorbar(L, np.mean(f), np.sqrt(f.var()), fmt='o', linestyle='None', color='r')
        plt.errorbar(L, np.mean(f_HS), np.sqrt(f_HS.var()), fmt='o', linestyle='None', color='y')#, label='Harmonic space estimator from maps')

"""
#end comp
#plt.plot(L_m_map3, ff_m_map3/ff_m_map3[8]*ff_H_a[4], 'D', label='map fov=40 nside=2048 amp=10/k')# n=15')
#plt.plot(L_m_map5, ff_m_map5/ff_m_map5[3]*ff_H_a[4], 'o', label='map fov=40 nside=2048 amp=1/k')

    

plt.xlim(0,3000)
if expt=='planck':
    plt.ylim(-0.3,1.2)
    plt.legend(loc='center right')#, bbox_to_anchor=(0.5, 0.35))
else:
    plt.ylim(0,7)
    plt.legend(loc='upper center')#, bbox_to_anchor=(0.5, 0.35))
plt.ylabel('Multiplicative bias')
plt.xlabel('Lensing wavenumber L')
plt.title('Response to dilatation of temperature maps for '+expt)
plt.show()




k_list=np.array([1,5,10,20, 30, 50,70, 90])
theta_max=(np.pi/180.)*40.
renorm=2.*np.pi/theta_max
L_maps_shear=renorm*k_list
ff_maps_shear_RS=np.zeros((len(k_list),))
ff_maps_shear_HS=np.zeros((len(k_list),))
sigma_maps_shear_RS=np.zeros((len(k_list),))
sigma_maps_shear_HS=np.zeros((len(k_list),))

est='shear_plus'
dir='/Users/Heather/Documents/Code/form_factor_package_Martin'
for i,k in enumerate(k_list):
    f=np.loadtxt(dir+'/Heather/form_factor_results_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_lots')
    f_null=np.loadtxt(dir+'/Heather/form_factor_results_null_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_lots')
    f-=f_null
    f=f[0:n_iter]
    f_HS=np.loadtxt(dir+'/Heather/form_factor_results_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_HS_lots')
    f_null_HS=np.loadtxt(dir+'/Heather/form_factor_results_null_'+spec+'_'+est+'_k'+str(int(k))+'_amp1_HS_lots')
    f_HS-=f_null_HS
    f_HS=f_HS[0:n_iter]
    
    ff_maps_shear_RS[i]=np.mean(f)
    ff_maps_shear_HS[i]=np.mean(f_HS)
    
    sigma_maps_shear_RS=np.sqrt(f.var())
    sigma_maps_shear_HS=np.sqrt(f_HS.var())

plt.errorbar(L_maps_shear,ff_maps_shear_RS*(-1)*A1,sigma_maps_shear_RS*A1,fmt='o',linestyle='None')
#plt.errorbar(L_maps_shear,ff_maps_shear_HS,sigma_maps_shear_HS,fmt='o',linestyle='None')
plt.xlabel('wave number L')
plt.ylabel('Amplitude F(L)')
plt.title('Form factor for tt shear plus')
#plt.savefig('form_factor_heather_'+exp+'_'+spec+'_'+est+'.pdf')#ACT_dilatation_l20000_finer.pdf")
plt.show()


"""
plt.plot(L_H_a[1:], (ff_m_map/ff_m_map[0])/ff_H_a[1:])
plt.title('Martin ff from maps / Heather analytical ff')
plt.show()"""


















