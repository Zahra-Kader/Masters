# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:46:35 2018

@author: KaderF
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import useful_functions as uf
import distance as cd
import density as den
import constants as cc
import reionization as cr
from decimal import Decimal

###print (chi_m)
cosmo=uf.cosmo
Mps_interpf=uf.Mps_interpf

n_points=2000
z_r=6.' n0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g #in units of g/Mpc^3
sigma_T=cc.sigma_T_Mpc
tau_r=0.055
T_rad=2.73 #In Kelvin

chi_r = uf.chi(z_r)
chi_m = uf.chi(1100)
chi_min=0.0001
k_min=0.0001
k_max=10.
kpar=np.linspace(k_min,k_max,n_points)
chi_array=np.linspace(chi_min,chi_m,n_points)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
#get matter power spec data


f=uf.f(z_r)
#plt.loglog(uf.kabs,Mps_interpf(uf.kabs))
#plt.show()

const=(f*uf.H0*2*np.pi)/cc.c_light_Mpc_s*T_rad
'''
delta_chi=chi_m-1.
g=((1.-np.exp(-tau_r))/np.sqrt(2*np.pi*(delta_chi)**2))*np.exp(-(chi-chi_r)**2/(2*(delta_chi)**2))
Integral_chip=sp.integrate.trapz(g,chi)
##print (Integral_chip,'chip')
g=g/chi**2
Integral_chi=sp.integrate.trapz(g,chi)
##print (Integral_chi,'chi')
'''
def g(chi):
        delta_chi=chi_m-chi_min
        g=((1.-np.exp(-tau_r))/np.sqrt(2*np.pi*(delta_chi)**2))*np.exp(-(chi-chi_r)**2/(2*(delta_chi)**2))
        return g

'Finding integral of MPS/kpar**2 using scipy.integrate.trapz'
def Mpspec():
    Mpspec=(Mps_interpf(kpar))/kpar**2
    Integral=sp.integrate.trapz(Mpspec,kpar)
    return Integral
##print ('%.2E' % Decimal(Mpspec()))
#Cl_limit=const**2*(Mpspec()*Integral_chip*Integral_chi)
###print (Cl_limit)
#g_norm=sp.integrate.quad(g,0,np.inf)

###print (g_norm) 
###print (1.-np.exp(-tau_r)) 
#plt.loglog(chi,g(chi))
#plt.xlabel('chi (Mpc)')
#plt.ylabel(r'g (chi) ($\rm{Mpc}^{-1}$)')
#plt.xlim(10,4e4)
##plt.show()
##plt.loglog(chi,g(chi)*chi**(-2.),'g')
#plt.show()

#function to perform triple integral
def Cl_all_ell(ell):
    kpar=np.logspace(np.log(k_min),np.log(10.),n_points)
    #kpar = np.logspace(-4.,1.,n_points)
    Cl=np.array([])
    for i in ell:
        integrand=[const**2*g(chi_r)**2*(kpar/(kpar**2+(i**2/chi_r**2)))**2/chi_r**2 * Mps_interpf(np.sqrt((i**2/chi_r**2) + kpar**2. ))]
        Cl=np.append(Cl,sp.integrate.trapz(integrand,kpar))
    return Cl

def Cl_single_ell(ell):
    kpar=np.logspace(np.log(k_min),np.log(10.),n_points)
    #kpar = np.logspace(-4.,1.,n_points)
    integrand=[const**2*g(chi_r)**2*(kpar/(kpar**2+(ell**2/chi_r**2)))**2/chi_r**2 * Mps_interpf(np.sqrt((ell**2/chi_r**2) + kpar**2. ))]
    Cl=sp.integrate.trapz(integrand,kpar)
    return Cl
ell=np.linspace(0,2000,n_points)

###print(g(chi_r)**2/(chi_r**2))

Mps=Mps_interpf(kpar)
##plt.loglog(Mps)
##plt.show()
##print (len(Mps))
spacing=5
n=n_points/spacing
Mps_spacing=Mps[0::spacing]
##print (Mps_spacing)
kpar_sq_inv=kpar**(-2)
kpar_sq_inv_spacing=kpar_sq_inv[0::spacing]
##print (kpar_sq_inv_spacing)
b=Mps_spacing*kpar_sq_inv_spacing
Trapz=(k_max-k_min)/n*(sum(b)-b[0]/2-b[-1]/2)
##print ('%.2E' % Decimal(Trapz),'numerical integ of P(kpar)/kpar**2 with 20 samples')
##print (const**2*g(chi_r)**2*Trapz/chi_r**2,'total numerical integ value with constants')
##print (Cl_single_ell(ell=0),'actual Cl value at ell=0')
##print (Cl_single_ell(ell=10),'actual Cl value at ell=10')
###print (Cl_all_ell(ell))

plt.loglog(ell,ell*(ell+1)*Cl_all_ell(ell)/(2*np.pi)*1e12,'b')
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)C_l^{pp}(l)/2\pi [\mu K^2]}$')
plt.loglog(ell,ell*(ell+1)*Cl_single_ell(ell=0)/(2*np.pi)*1e12,'r')
blue_patch = mpatches.Patch(color='blue', label=r'all $\rm{l}$')
red_patch = mpatches.Patch(color='red', label=r'low $\rm{l}$')
plt.legend(handles=[blue_patch,red_patch])
plt.show()
