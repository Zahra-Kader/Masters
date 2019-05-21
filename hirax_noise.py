import numpy as np
import matplotlib.pyplot as plt
import useful_functions as uf
import sys
sys.path.insert(0,'/home/zahra/hirax_tools/hirax_tools')
sys.path.insert(0,'/home/zahra/python_scripts/CMB_noise')
import scal_power_spectra as cmb
from array_config import HIRAXArrayConfig
from neutral_H_autocorr import Cl_21_func_of_y,Cl_21,Integrand_doppler_21cm
import scipy as sp
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
import pylab
import constants as cc

from lin_kSZ_power_spec import interp_OV_full_signal
from bispec import crosscorr_squeezedlim

'''run in python 3'''
chi=uf.chi
r=uf.r
nu_min=585. #freq in MHz, these min and max values are based on HIRAX
nu_max=705.
D=uf.D_1
H=uf.H
f=uf.f
c=cc.c_light_Mpc_s

def HiraxNoise(l,Ddish,Dsep,zi): #the Dsep refers to the dish seperation including the dish diameter of 6. So we're assuming dishes are 1m away

        cp= HIRAXArrayConfig.from_n_elem_compact(1024,Dsep)
        nubar=(nu_min+nu_max)/2.

        Fov_deg=(cp.fov(frequency=nubar)) * ((180./np.pi)**2)
        Fov_str=(cp.fov(frequency=nubar))
        Tsys= 50. + 60.*((nubar/300.)**-2.5)
        lam=3.e8/(nubar*1.e6)
        Nbeam=1.
        npol=2.
        nu21=1420.e6
        Aeff=np.pi*((Ddish/2.)**2)*0.67
        Ttot=100.8e6 #36e6   4*365*24*3600.
        #Ttot=2*365*24*3600
        Sarea=15000.
        pconv=(chi(zi)**2)*r(zi)
        n_u=cp.baseline_density_spline(frequency=nubar)
        n=np.array([])
        for i in l:
            if n_u(i/(2.*np.pi))==0:
                n_u1=1/1e100
            else:
                n_u1=n_u(i/(2.*np.pi))
            n=np.append(n,n_u1)
        Nbs= 1024.*(1024.-1.)
        #norm=Nbs/sp.integrate.trapz(n*2*np.pi*(l/(2.*np.pi)), l/(2.*np.pi))
        C=n#*norm
        A_bull= ((Tsys**2)*(lam**4)*Sarea) / (nu21*npol*Ttot*(Aeff**2)*Fov_deg*Nbeam)
        Warren=Tsys**2*lam**2/Aeff*4.*np.pi/(nu21*npol*Nbeam*Ttot)
        #return (Warren/C)*1e12
        return (A_bull/C)*1e12

ell_large=np.linspace(10,2000,100000)
y_large=np.linspace(1,3000,100000)

n=1000
ell=np.linspace(10,5000,n)
y=np.linspace(200,5000,n)


#np.savetxt('Hirax_noise_z_1_Ddish_6_Dsep_7_geom.out',(ell_large,HiraxNoise(ell_large,6.,7.,1.)))
#ell_new,Hirax_noise_z_1=np.genfromtxt('/home/zahra/python_scripts/kSZ_21cm_signal/Hirax_noise_z_1_Ddish_6_Dsep_7.out')
Hirax_noise_z_1pt27_interp = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.27), bounds_error=False)



def Hirax_noise_21cm_vel(ell,y):
    z=1.
    k=np.sqrt(y**2/r(z)**2+ell**2/chi(z)**2)
    mu_k=(y/r(z))/k
    v_expression_dimless=1/c*f(z)*H(z)/(1+z)*mu_k/k ##took out the D^2 factor-don't know if I need it
    return Hirax_noise_z_1_interp(ell)*v_expression_dimless**2

def Func_2d(ell,z,y,Func):
    Func_mat=np.zeros((len(ell),len(y)))
    for i in range(len(ell)):
        for j in range(len(y)):
            Func_ind=Func(ell[i],z,y[j])
            Func_mat[i][j]=Func_ind
    return Func_mat


def Cl_vel(ell,y): ###Just a check that this is the same as the Integrand_doppler_21cm function, which it now is
    z=1.
    k=np.sqrt(y**2/r(z)**2+ell**2/chi(z)**2)
    mu_k=(y/r(z))/k
    v_expression_dimless=1./c*f(z)*H(z)/(1+z)*mu_k/k ##took the D^2 factor out because the 21cm density Cl has this factor already
    print (v_expression_dimless,'v_fac')
    return Cl_21_func_of_y(ell,y)*v_expression_dimless**2

#print (Cl_vel(2000,3000))

def HI_den_SNR(ell,z,y,ell_2d):
    S=Func_2d(ell,z,y,Cl_21_func_of_y)
    N=Hirax_noise_z_1pt27_interp(ell_2d)
    #N=HiraxNoise(ell_2d,6.,7.,z)
    sigma=S+N
    return S/sigma

def HI_vel_SNR(ell,y,ell_2d):
    S=Func_2d(ell,z,y,Integrand_doppler_21cm)
    N=Func_2d(ell,y,Hirax_noise_21cm_vel)
    sigma=S+N
    return S/sigma

cmb_spec=cmb.cl_tt_func_bis
cmb_noise=cmb.nl_tt_ref_func_bis


def Bispec_SNR(ell,y,ell_2d):
    S = Func_2d(ell,y,crosscorr_squeezedlim)
    N_ksz=cmb_spec(ell_2d)+cmb_noise(ell_2d)
    S_ksz=interp_OV_full_signal(ell_2d)
    S_21_den=Func_2d(ell,y,Cl_21_func_of_y)
    S_21_vel=Func_2d(ell,y,Integrand_doppler_21cm)
    N_21_den=Hirax_noise_z_1_interp(ell_2d)
    N_21_vel=Func_2d(ell,y,Hirax_noise_21cm_vel)
    sigma=(S_ksz+N_ksz)*(N_21_den+N_21_vel+S_21_den+S_21_vel)+3.*S**2
        #y_ind=np.linspace(y[j],y[j+1],n_new)
    #Ell,Y=np.meshgrid(ell_ind,y_ind)
    return S/sigma

ell_ind=592.
kperp=ell_ind/chi(1)
y_ind=581.
kpar=y_ind/r(1)
'''
print (Func_2d(ell,y,crosscorr_squeezedlim).max(),'bispec_signal')
print (cmb_spec(ell).max(),'cmb_spec')
print (cmb_noise(ell).max(),'cmb_noise')
print (interp_OV_full_signal(ell).max(),'OV_signal')
print (Func_2d(ell,y,Cl_21_func_of_y).max(),'S_den')
print (Func_2d(ell,y,Integrand_doppler_21cm).max(),'S_vel')
print (Hirax_noise_z_1_interp(ell).max(),'N_den')
print (Func_2d(ell,y,Hirax_noise_21cm_vel).max(),'N_vel')
print (HI_den_SNR(ell,y,ell).max(),'HI_den_snr')
print (Bispec_SNR(ell,y,ell).max(),'SNR')


#######################################################

#CHECKING ORDER OF MAGNITUDES FOR WORK WRITE UP

Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
fsky=Sarea/4./np.pi
delta_nu_dimless=400./1420.
Mode_Volume_Factor=fsky*Sarea*delta_nu_dimless#*2
const=Mode_Volume_Factor*0.5/4./np.pi**2
br1=interp_OV_full_signal(ell_ind)+cmb_spec(ell_ind)+cmb_noise(ell_ind)
print (interp_OV_full_signal(ell_ind),'kSZ_signal')
print (cmb_spec(ell_ind),'cmbspec')
print (cmb_noise(ell_ind),'cmbnoise')
br2=Hirax_noise_z_1_interp(ell_ind)+Hirax_noise_21cm_vel(ell_ind,y_ind)+Cl_21_func_of_y(ell_ind,y_ind)+Integrand_doppler_21cm(ell_ind,y_ind)
print (Hirax_noise_z_1_interp(ell_ind),'21_den_noise')
print (Hirax_noise_21cm_vel(ell_ind,y_ind),'21_vel_noise')
print (Cl_21_func_of_y(ell_ind,y_ind),'21_den_sig')
print (Integrand_doppler_21cm(ell_ind,y_ind),'21_vel_signal')
t1=br1*br2
t2=3*crosscorr_squeezedlim(ell_ind,y_ind)**2
var=t1+t2
print (var)
S=crosscorr_squeezedlim(ell_ind,y_ind)
print (S,'bispec_signal')
bin_widths=0.01**2*r(1)*chi(1)
SNR=np.sqrt(const*bin_widths*S**2/var*ell_ind)
print (SNR,kperp,kpar)

S_21_vel=Integrand_doppler_21cm(ell_ind,y_ind)
N_21_vel=Hirax_noise_21cm_vel(ell_ind,y_ind)
sigma_vel=S_21_vel+N_21_vel
SNR=np.sqrt(const*bin_widths*S_21_vel**2/sigma_vel**2*ell_ind)
print (SNR,'21_vel')

S_21_den=Cl_21_func_of_y(ell_ind,y_ind)
N_21_den=Hirax_noise_z_1_interp(ell_ind)
sigma_den=S_21_den+N_21_den
SNR=np.sqrt(const*bin_widths*S_21_den**2/sigma_den**2*ell_ind)
print (SNR,'21_den')
######################################################################

plt.plot(ell,Cl_vel(ell,72),'b')
plt.plot(ell,Integrand_doppler_21cm(ell,1,72),'g')
plt.show()

#print (ell)
#print (Hirax_noise_21cm_vel(ell,72.))
plt.loglog(ell,Hirax_noise_21cm_vel(ell,72.))
plt.loglog(ell,Hirax_noise_21cm_vel(ell,572.))
plt.loglog(ell,Hirax_noise_21cm_vel(ell,1072.))
plt.loglog(ell,Hirax_noise_21cm_vel(ell,2072.))
plt.loglog(ell,Hirax_noise_21cm_vel(ell,2572.))
plt.loglog(ell,Hirax_noise_z_1_interp(ell),'k')
plt.legend(('y=72','y=572','y=1072','y=2072','y=2572'))
plt.xlabel('l')
plt.ylabel(r'$\rm{N_l^{HI}(y,z=1)/\sqrt{f_{sky}l [\mu K^2]}$')
plt.xlim(100,5000)
plt.ylim(1e-16,1)
plt.show()
#print (Cl_21_func_of_y(1,277,ell)+HiraxNoise(ell,6,7,1))
#plt.plot(ell,HiraxNoise(ell,6,7,1))
#plt.plot(ell,Hirax_noise_z_1_interp(ell))
#plt.show()
'''




#pylab.pcolormesh(Bispec_2d_interp(ell,y))
#pylab.show()
#Cl_21_2d_interp = sp.interpolate.interp2d(ell, y, Cl_21_2d(ell,1,y)) #get same result from using mesh grid

def SNR_binned(z,SNR):
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    Mode_Volume_Factor=fsky*Sarea*delta_nu_dimless#*2
    num_k_bins_y=15
    num_k_bins_ell=15
    n=1000
    kpar_arr = np.zeros(num_k_bins_y) ; kperp_arr= np.zeros(num_k_bins_ell) ; SNR_arr = np.zeros((num_k_bins_y,num_k_bins_ell))
    kpar_min = 0.02
    kperp_min = 0.02
    delta_kpar = 0.01
    delta_kperp = 0.01
    y_min=kpar_min*r(z)
    ell_min=kperp_min*chi(z)
    delta_y=delta_kpar*r(z)
    delta_ell=delta_kperp*chi(z)
    y_arr=np.linspace(y_min,y_min+num_k_bins_y*delta_y,n)
    ell_arr=np.linspace(ell_min,ell_min+num_k_bins_ell*delta_ell,n)
    for bin_number_y in np.linspace(0,num_k_bins_y-1,num_k_bins_y):
        kpar_bin_min, kpar_bin_max = kpar_min + delta_kpar*np.array([bin_number_y, bin_number_y+1])    # bin_number starts at 0
        #print (kpar_bin_min*r(z),kpar_bin_max*r(z),'kpar')
        for bin_number_ell in np.linspace(0,num_k_bins_ell-1,num_k_bins_ell):
            kperp_bin_min, kperp_bin_max = kperp_min + delta_kperp*np.array([bin_number_ell, bin_number_ell+1])
            #print (kperp_bin_min*r(z),kperp_bin_max*r(z),'kperp')
            kpar_arr[np.int(bin_number_y)] = kpar_bin_min  ; kperp_arr[np.int(bin_number_ell)] = kperp_bin_min
            y_bin_arr = y_arr[(y_arr > kpar_bin_min*r(z)) & (y_arr < kpar_bin_max*r(z))]
            ell_bin_arr = ell_arr[(ell_arr > kperp_bin_min*chi(z)) & (ell_arr < kperp_bin_max*chi(z))]
            ell_bin_2d_arr=np.outer(ell_bin_arr,np.ones(len(y_bin_arr)))
            #SN_ratio_2d_arr = SNR(ell_bin_arr,y_bin_arr) #SNR(Ell,Y)
            SN_ratio_2d_arr=SNR(ell_bin_arr,z,y_bin_arr,ell_bin_2d_arr)
            integrand=SN_ratio_2d_arr**2*ell_bin_2d_arr
            int1=sp.integrate.trapz(integrand, ell_bin_arr, axis=0)
            SNR_sq = 0.5 * Mode_Volume_Factor/(4.*np.pi**2) * sp.integrate.trapz(int1, y_bin_arr, axis=0)
            SNR_arr[np.int(bin_number_y),np.int(bin_number_ell)] = np.sqrt(SNR_sq)
                # print 'kparmin=', kpar_bin_min, 'kperpmin=', kperp_bin_min, np.sqrt(SNR_sq)
    return kperp_arr,kpar_arr,SNR_arr



kperp_arr=SNR_binned(1.27,HI_den_SNR)[0]
kpar_arr=SNR_binned(1.27,HI_den_SNR)[1]
SNR_arr=SNR_binned(1.27,HI_den_SNR)[2]

pylab.pcolormesh(kperp_arr,kpar_arr,SNR_arr) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
pylab.xlim([np.min(kperp_arr),np.max(kperp_arr)]) ; pylab.ylim([np.min(kpar_arr),np.max(kpar_arr)])
plt.xlabel(r'$k_\perp$',fontsize=12); plt.ylabel(r'$k_\parallel$',fontsize=12); plt.title('Pixel SN for 21-21', x=1.13, y=1.05)
pylab.show()
'''
def SNR_integrand_mat(ell,z_i,y,S):
    kperp=ell/chi(z_i)
    kpar=y/r(z_i)
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2.*np.pi)**2*Sarea*delta_nu_dimless
    integrand_arr=np.array([])
    for i in ell:
        for j in y:
            cl_21=S(i,z_i,j)
            integrand=i*(cl_21)**2/(cl_21+Hirax_noise_z_1_interp(i))**2
            integrand_arr=np.append(integrand_arr,integrand)
    integrand_mat=np.reshape(integrand_arr,(n,n))
    return np.sqrt(const*integrand_mat),kperp,kpar


kperp=SNR_integrand_mat(ell,1,y,Cl_21_func_of_y)[1]
kpar=SNR_integrand_mat(ell,1,y,Cl_21_func_of_y)[2]
pylab.pcolormesh(kperp,kpar,SNR_integrand_mat(ell,1,y,Cl_21_func_of_y)[0]) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel(r'$k_{\perp}$')
plt.ylabel(r'$k_{\parallel}$')
plt.title('Differential S/N')
plt.show()

def SNR_tot(ell,z_i,y):
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2.*np.pi)**2*Sarea*delta_nu_dimless
    Ell,Y=np.meshgrid(ell,y)
    integrand=Ell*(Cl_21_func_of_y(Ell,z_i,Y))**2/(Cl_21_func_of_y(Ell,z_i,Y)+Hirax_noise_z_1_interp(Ell))**2
    #integral=sp.integrate.cumtrapz(integrand,ell,initial=0)
    integral=sp.integrate.trapz(sp.integrate.trapz(integrand,ell,axis=0),y,axis=0)
    integral=const*integral
    return np.sqrt(integral)

#print (SNR_tot(ell,1,y))



def SNR(ell,z_i,y,S):
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2.*np.pi)**2*Sarea*delta_nu_dimless
    #snr_mat=np.zeros((n-1,n-1))
    integral_tot=np.zeros((n-1,n-1))
    n_new=10
    for i in range(n-1):
        ell_ind=np.linspace(ell[i],ell[i+1],n_new)
        for j in range(n-1):
            y_ind=np.linspace(y[j],y[j+1],n_new)
            Ell,Y=np.meshgrid(ell_ind,y_ind)
            cl_21 = S(ell_ind,y_ind)#S(Ell,z_i,Y)
            nl_21=Hirax_noise_z_1_interp(Ell)
            #nl_21 = np.reshape(HiraxNoise(Ell,6,7,1),(n_new,n_new))
            integrand=Ell*(cl_21**2.)/(cl_21+nl_21)**2
            integral=sp.integrate.trapz(sp.integrate.trapz(integrand,y_ind,axis=0),ell_ind,axis=0)
            integral_tot[i][j]=integral
            integral_tot=np.sqrt(const*integral_tot)
    #integral_tot=np.flip(integral_tot,0)
    return integral_tot

#print (SNR(ell,1,y,Cl_21_func_of_y))

#pylab.pcolormesh(SNR(ell,1,y,Cl_21_2d_interp)) ;  cbar=plt.colorbar()
#plt.show()
Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
fsky=Sarea/4./np.pi
delta_nu_dimless=400./1420.
const=2.*fsky/(2*np.pi)**2*Sarea*delta_nu_dimless
integrand=Ell*Cl_21_func_of_y(Ell,1,Y)**2/(Cl_21_func_of_y(Ell,1,Y)+Hirax_noise_z_1_interp(Ell))**2
integral=np.trapz(np.trapz(integrand,ell,axis=0),y,axis=0)
#print (np.sqrt(const*integral))

def SNR_area(ell,z_i,y,S):
    #n=100
    #kperp=np.linspace(0.01,0.2,n)
    #kpar=kperp
    kperp=ell/chi(z_i)
    kpar=y/r(z_i)
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2*np.pi)**2*Sarea*delta_nu_dimless
    snr_mat=np.zeros((n-1,n-1))
    for i in range(n-1):
        nl_21 = Hirax_noise_z_1_interp(ell[i+1])
        #ell_ind=np.linspace(ell[i],ell[i+1],n_new)
        for j in range(n-1):
            #y_ind=np.linspace(y[j],y[j+1],n_new)
            #Ell,Y=np.meshgrid(ell_ind,y_ind)
            area=(ell[i+1]-ell[i])*(y[j+1]-y[j])
            #print (area)
            cl_21 = S(ell[i+1],z_i,y[j+1])
            integrand=ell[i+1]*(cl_21**2.)/(cl_21+nl_21)**2
            snr_mat[i][j]=integrand*area
            #integral=sp.integrate.trapz(sp.integrate.trapz(integrand,ell_ind,axis=0),y_ind,axis=0)
            #integral_tot=np.append(integral_tot,integral)
            #integral_tot=np.sqrt(const*integral_tot)
    #integral_tot=np.flip(integral_tot,0)
    return np.sqrt(const*snr_mat),kperp,kpar

SNR_area_21_den=SNR_area(ell,1,y,Cl_21_func_of_y)[0]

N, M = SNR_area_21_den.shape
div=10
assert N % div == 0
assert M % div == 0
A1 = np.zeros((N//div, M//div))
for i in range(N//div):
    for j in range(M//div):
         A1[i,j] = np.mean(SNR_area_21_den[2*i:2*i+2, 2*j:2*j+2])

#print (np.sum(SNR_area(ell,1,y,Cl_21_func_of_y)[0]))

kperp=SNR_area(ell,1.,y,Cl_21_func_of_y)[1]#[::div]
kpar=SNR_area(ell,1.,y,Cl_21_func_of_y)[2]#[::div]
#pylab.pcolormesh(kperp,kpar,SNR_area_21_den) ;  cbar=plt.colorbar()
#plt.show()

def SNR_bispec_integrand(ell,z_i,y,delta_z):
    kperp=ell/chi(z_i)
    kpar=y/r(z_i)
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2.*np.pi)**2*Sarea*delta_nu_dimless
    integrand_arr=np.array([])
    for i in ell:
        for j in y:
            S_bispec = crosscorr_squeezedlim(i,z_i,j,delta_z)
            N_ksz=cmb_spec(i)+cmb_noise(i)
            S_ksz=interp_OV_full_signal(i)
            S_21_den=Cl_21_func_of_y(i,z_i,j)
            S_21_vel=Integrand_doppler_21cm(i,z_i,j)
            N_21=Hirax_noise_z_1_interp(i)
            sigma=(S_ksz+N_ksz)*(2.*N_21+S_21_den+S_21_vel)+3.*S_bispec**2
            #y_ind=np.linspace(y[j],y[j+1],n_new)
            #Ell,Y=np.meshgrid(ell_ind,y_ind)
            integrand=i*(S_bispec**2.)/(sigma)**2
            integrand_arr=np.append(integrand_arr,integrand)
    integrand_mat=np.reshape(integrand_arr,(n,n))
    return np.sqrt(const*integrand_mat),kperp,kpar


def SNR_bispectrum(ell,z_i,y,delta_z):
    kperp=ell/chi(z_i)
    kpar=y/r(z_i)
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2.*np.pi)**2*Sarea*delta_nu_dimless
    snr_mat=np.zeros((n-1,n-1))
    n_new=100
    for i in range(n-1):
        #ell_ind=np.linspace(ell[i],ell[i+1],n_new)
        for j in range(n-1):
            S_bispec = crosscorr_squeezedlim(ell[i+1],z_i,y[j+1],delta_z)
            N_ksz=cmb_spec(ell[i+1])+cmb_noise(ell[i+1])
            S_ksz=interp_OV_full_signal(ell[i+1])
            S_21_den=Cl_21_func_of_y(ell[i+1],z_i,y[j+1])
            S_21_vel=Integrand_doppler_21cm(ell[i+1],z_i,y[j+1])
            N_21=Hirax_noise_z_1_interp(ell[i+1])
            sigma=(S_ksz+N_ksz)*(2.*N_21+S_21_den+S_21_vel)+3.*S_bispec**2
            #y_ind=np.linspace(y[j],y[j+1],n_new)
            #Ell,Y=np.meshgrid(ell_ind,y_ind)
            area=(ell[i+1]-ell[i])*(y[j+1]-y[j])
            #print (area)

            integrand=ell[i+1]*(S_bispec**2.)/(sigma)**2
            snr_mat[i][j]=integrand*area
            #integral=sp.integrate.trapz(sp.integrate.trapz(integrand,ell_ind,axis=0),y_ind,axis=0)
            #integral_tot=np.append(integral_tot,integral)
            #integral_tot=np.sqrt(const*integral_tot)
    #integral_tot=np.flip(integral_tot,0)
    return np.sqrt(const*snr_mat),kperp,kpar

kperp=SNR_bispectrum(ell,1.,y,0.3)[1]
kpar=SNR_bispectrum(ell,1.,y,0.3)[2]
pylab.pcolormesh(kperp,kpar,SNR_bispec_integrand(ell,1.,y,0.3)[0]) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel(r'$k_{\perp}$')
plt.ylabel(r'$k_{\parallel}$')
plt.title('Differential S/N')
plt.show()

#print (np.sum(SNR_area(ell,1,y,Integrand_doppler_21cm)[0]),'sum_snr_area')
kperp=SNR_area(ell,1,y,Cl_21_func_of_y)[1]
kpar=SNR_area(ell,1,y,Cl_21_func_of_y)[2]
pylab.pcolormesh(kperp,kpar,SNR_area(ell,1,y,Cl_21_func_of_y)[0]) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel(r'$k_{\perp}$')
plt.ylabel(r'$k_{\parallel}$')
plt.title('S/N')
plt.show()


#print (np.sum(SNR(ell,1,y)),'sum')



#plt.ylim(1e-20,1e-7)
#plt.xlim(100,5000)
#plt.show()

#SNR=Cl_21(1,ell)/HiraxNoise(ell,6,7,1)
#print (HiraxNoise(ell,6.,20.,2.))
#plt.ylim(1e-10,1)
#plt.ylim(1,100)

ell=np.linspace(50,5000,100)
plt.loglog(ell,1.e-6*Integrand_doppler_21cm(ell,z1=1,y=72.),'b')
plt.loglog(ell,1.e-6*Integrand_doppler_21cm(ell,1,572.),'g')
plt.loglog(ell,1.e-6*Integrand_doppler_21cm(ell,1,1072.),'r')
plt.loglog(ell,1.e-6*Integrand_doppler_21cm(ell,1,1572.),'m')
plt.loglog(ell,1.e-6*Integrand_doppler_21cm(ell,1,2072.),'k')


plt.imshow((ell,SNR(ell,1,y)), cmap='hot', interpolation='nearest')

plt.plot(ell,1e-6*Cl_21_func_of_y(ell,72))
plt.plot(ell,1e-6*Cl_21_func_of_y(ell,572))
plt.plot(ell,1e-6*Cl_21_func_of_y(ell,1072))
plt.plot(ell,1e-6*Cl_21_func_of_y(ell,1572))
plt.plot(ell,1e-6*Cl_21_func_of_y(ell,2072))


plt.legend(('y=72','y=572','y=1072','y=1572','y=2072'))
plt.loglog(ell,1e-6*HiraxNoise(ell,6.,7.,1)/np.sqrt(ell*0.36),'--')
plt.loglog(ell,1e-6*Hirax_noise_z_1_interp(ell)/np.sqrt(ell*0.36),'--')

#plt.loglog(ell,Hirax_noise_z_1_interp(ell))
plt.xlabel('l')
#plt.ylabel('S/N')
plt.xlim(50,5e3)
plt.ylim(1e-12,1e-8)
plt.ylabel(r'$\rm C_l^{HI}(y,z=1)$ vs $N_l^{HI}(y,z=1)/\sqrt{f_{sky}l} [mK^2]$')

#plt.legend(('HI density power spectrum','HIRAX noise'))
plt.show()
'''
