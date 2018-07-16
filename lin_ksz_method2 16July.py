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
z_r=10.
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
k_max=0.1
chi_array=np.linspace(chi_min,chi_m,n_points)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
#get matter power spec data


f=uf.f(z_r)
#plt.loglog(uf.kabs,Mps_interpf(uf.kabs))
#plt.show()

const=(f*uf.H(z_r))/cc.c_light_Mpc_s*T_rad
z=uf.z
zed=uf.zed
'''
delta_chi=chi_m-1.
g=((1.-np.exp(-tau_r))/np.sqrt(2*np.pi*(delta_chi)**2))*np.exp(-(chi-chi_r)**2/(2*(delta_chi)**2))
Integral_chip=sp.integrate.trapz(g,chi)
##print (Integral_chip,'chip')
g=g/chi**2
Integral_chi=sp.integrate.trapz(g,chi)
##print (Integral_chi,'chi')
'''
def y(z):
    y=(1+z)**(1.5)
    return y

def x(z):
    delta_z=0.5
    delta_y=1.5*(np.sqrt(1+z_r))*delta_z
    arg=(y(z_r)-y(z))/delta_y
    x=0.5*(1+np.tanh(arg))
    return x

def g(inp):
    #tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    #tau_der=np.abs(cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo))[0]
    N_He=2.
    Y=0.24
    n_e=(1.-(4.-N_He)*Y/4.)*cosmo['omega_b_0']*rho_c/cc.m_p_g
    #tau_der=(1.+zed(chi))**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g                                                                                                                                     
    #g=uf.f(z)*H0*2*np.pi*tau_der*np.exp(-tau)/cc.c_light_Mpc_s
    g=sigma_T*n_e*(1+inp)**2*x(inp)                                                     
    return g
'''
def g(chi):
        delta_chi=chi_m-chi_min
        g=((1.-np.exp(-tau_r))/np.sqrt(2*np.pi))*np.exp(-(chi-chi_r)**2/(2))
        return g
'''
def u(inp):
    u=g(inp)/((1+inp)*x(inp))
    return u
'''
plt.plot(z,g(z))
plt.show()
#plt.plot(z,u(z))
g_norm=sp.integrate.quad(g,-np.inf,np.inf)[0]
#u_norm=sp.integrate.quad(u,-np.inf,np.inf)[0]

print (g_norm)
plt.plot(z,g(z)/g_norm)
plt.xlabel('z')
plt.ylabel('g(z)')
plt.xlim(0.01,100)
plt.show()
g_norm1=1/g_norm*sp.integrate.quad(g,-np.inf,np.inf)[0]
print (g_norm1)
#plt.plot(chi_array,g_instant(zed(chi_array)))
'''
'Finding integral of MPS/kpar**2 using scipy.integrate.trapz'

#def Mpspec():
   # Mpspec=(Mps_interpf(kpar))/kpar**2
   # Integral=sp.integrate.trapz(Mpspec,kpar)
   # return Integral
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
    #kpar = np.logspace(-4.,1.,n_points)
    Cl=np.array([])
    for i in ell:
        kpar=np.linspace(k_min,k_max,n_points)
        k=np.sqrt(kpar**2+i**2/chi_r**2)
        j=np.sqrt(np.pi/(2*k*chi_r))*sp.special.jn(i+0.5,k*chi_r)
        #integrand=[g_norm**(2)*const**2 *kpar**2*Mps_interpf(k)/(k**4*chi_r**2)]
        integrand=[const**2 *u(z_r)**2*Mps_interpf(k)/(k**4*chi_r**2)]
        Cl=np.append(Cl,sp.integrate.trapz(integrand,kpar))
    return Cl

def Cl_single_ell(ell):
    #kpar = np.logspace(-4.,1.,n_points)
    Cl=np.array([])
    kpar=np.linspace(k_min,k_max,n_points)
    k=np.sqrt(kpar**2+ell**2/chi_r**2)
    #j=np.sqrt(np.pi/(2*k*chi_r))*sp.special.jn(i+0.5,k*chi_r)
    #integrand=[g_norm**(2)*const**2 *kpar**2*Mps_interpf(k)/(k**4*chi_r**2)]
    integrand=[const**2 *u(z_r)**2*Mps_interpf(kpar)/(kpar**4*chi_r**2)]
    Cl=np.append(Cl,sp.integrate.trapz(integrand,kpar))
    return Cl

###print(g(chi_r)**2/(chi_r**2))

ell=np.linspace(0,300,n_points)
print (Cl_single_ell(2))

'''
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
'''
plt.loglog(ell,ell**2*Cl_all_ell(ell)/(2*np.pi)*1e12,'b')
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)C_l^{pp}(l)/2\pi [\mu K^2]}$')
plt.loglog(ell,ell**2*Cl_single_ell(ell=0)/(2*np.pi)*1e12,'r')
blue_patch = mpatches.Patch(color='blue', label=r'all $\rm{l}$')
red_patch = mpatches.Patch(color='red', label=r'low $\rm{l}$')
plt.legend(handles=[blue_patch,red_patch])
plt.show()
