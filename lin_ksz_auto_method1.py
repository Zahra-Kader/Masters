# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:43:13 2018

@author: KaderF
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:01:45 2018

@author: KaderF
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import density as den
import constants as cc
import reionization as cr
import useful_functions as uf
#print (chi_m)

cosmo=uf.cosmo
zed=uf.zed
n_points=2000
H0=cc.H100_s*cosmo['h']
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
z_r=6.
chi_r=uf.chi(z_r)
chi_m = uf.chi(1100)
k_min=0.0001
k_max=10.
chi=np.linspace(0.0001,chi_r,n_points)
#chi = np.logspace(np.log(0.0001),np.log(chi_m),n_points)
kpar = np.logspace(-4.,1.,n_points)
Mps_interpf=uf.Mps_interpf
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]

#get matter power spec data
'''
tau=[]
tau_der_1=[]
tau_der_2=[]
z=np.linspace(0,1100,1101)
for i in z:
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau
plt.plot(tau)
plt.show()
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
plt.plot(z,tau_der_1,'b')
plt.plot(z,tau_der_2,'g')
plt.show()
'''
z=uf.z
def g(chi):
    tau=cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der=np.abs(cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo))[0]
    #tau_der=(1.+zed(chi))**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    g=uf.f(zed(chi))*H0*2*np.pi*tau_der*np.exp(-tau)/cc.c_light_Mpc_s
    return g
g_int = sp.interpolate.interp1d(chi,g(chi))

#g_norm=sp.integrate.quad(g,0,np.inf)
#print (g_norm)

def Cl_all_ell(ell):
    #chi = np.linspace(1.,chi_m,n_points)

    #kpar = np.linspace(0.0001,10.,n_points)
    Cl=np.array([])
    for i in ell:
        ch1,ch2=np.meshgrid(chi,chi)
        integrand=[g_int(ch1)*g_int(ch2)*(j/(j**2+(i**2/ch1**2)))**2*np.cos(kpar*(ch1-ch2))/ch1**2 * Mps_interpf(np.sqrt((i**2/ch1**2) + j**2. )) for j in kpar]
        integrand=[sp.integrate.trapz(sp.integrate.trapz(integrand,chi,axis=0)
                   ,chi,axis=0)]
        Cl=np.append(Cl,sp.integrate.trapz(integrand,kpar))
    return Cl

def Cl_low_ell():
    #chi = np.linspace(1.,chi_m,n_points)
    #kpar = np.linspace(0.0001,10.,n_points)
    ch1,ch2=np.meshgrid(chi,chi)
    integrand=[g_int(ch1)*g_int(ch2)*np.cos(kpar*(ch1-ch2))/ch1**2*Mps_interpf(j)/j**2 for j in kpar]
    integrand=[sp.integrate.trapz(sp.integrate.trapz(integrand,chi,axis=0)
               ,chi,axis=0)]
    Cl=np.abs(sp.integrate.trapz(integrand,kpar))
    return Cl 
    
ell=np.linspace(0,2000,n_points)

#print (Cl_all_ell(ell))

plt.loglog(ell,ell*(ell+1)*Cl_all_ell(ell)/(2*np.pi))
plt.loglog(ell,ell*(ell+1)*Cl_low_ell()/(2*np.pi))
plt.show()

