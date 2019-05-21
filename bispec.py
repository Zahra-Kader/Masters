import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import useful_functions as uf
import density as den
import constants as cc
from scipy.interpolate import interp1d
import pylab
'''changed the crosscorr_squeezedlim function to return a single ell i.e. took out the for loop'''

cosmo=uf.cosmo
Mps_interpf=uf.Mps_interpf
n_points=uf.n_points
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g #in units of g/Mpc^3
sigma_T=cc.sigma_T_Mpc
tau_r=0.055
z_r=10
T_rad=2.725 #In Kelvin

#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
#get matter power spec data
mu_e=1.14
m_p=cc.m_p_g #in grams
rho_g0=cosmo['omega_b_0']*rho_c
#plt.loglog(uf.kabs,Mps_interpf(uf.kabs))
#plt.show()
zed=uf.zed
z=uf.z
r=uf.r
chi=uf.chi

def ionized_elec(Yp,N_He):
    x=(1-Yp*(1-N_He/4))/(1-Yp/2)
    return x
x=ionized_elec(0.24,0)

n=5
ell=np.linspace(1,2000,n)
y=np.linspace(1,2000,n)
def crosscorr_squeezedlim(ell,y): #with the assumption that zi=z so no cos factor
    z_i=1.
    delta_z=0.3
    n=100
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp=np.linspace(1.e-4,10.,n)
    mu,kp=np.meshgrid(Mu,Kp)
    z=z_i
    T_mean_zi=uf.T_mean(z_i)
    chi_zi=chi(z_i)
    chi_z=chi(z)
    f_zi=uf.f(z_i)
    f_z=uf.f(z)
    D_zi=uf.D_1(z_i)
    r_zi=uf.r(z_i)
    D_z=uf.D_1(z)
    H_zi=uf.H(z_i)
    tau_z=uf.tau_inst(z)
    const=1.e6/(2.*np.pi)**2*T_rad*T_mean_zi/cc.c_light_Mpc_s*f_zi*D_zi**2*H_zi/(chi_zi**2*r_zi)/(1.+z_i)*x*(sigma_T*rho_g0/(mu_e*m_p))*delta_z*f_z*D_z**2*(1+z)*np.exp(-tau_z)
    #Cl=np.array([])
    kpar=y/r_zi
    k_perp=ell/chi_zi
    k=np.sqrt(k_perp**2+kpar**2)
    rsd=1.+f_zi*kpar**2/k**2
    theta_kp=kpar*mu/k+k_perp*np.sqrt(1.-mu**2)/k
    integrand=Mps_interpf(kp)*rsd*theta_kp**2*(Mps_interpf(k))#-mu*kp*np.gradient(Mps_interpf(k),axis=0))
    integral_sing=const*sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0)
    #Cl=np.append(Cl,integral)
    return integral_sing
