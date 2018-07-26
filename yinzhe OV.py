# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 19:29:37 2018

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

cosmo=uf.cosmo
Mps_interpf=uf.Mps_interpf

n_points=uf.n_points
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
k_min=1e-3
k_max=0.1
chi_array=np.linspace(chi_min,chi_m,n_points)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
#get matter power spec data
mu_e=1.44
m_p=cc.m_p_g #in grams
rho_g0=cosmo['omega_b_0']*rho_c
#plt.loglog(uf.kabs,Mps_interpf(uf.kabs))
#plt.show()

zed=uf.zed
z=uf.z

def ionized_elec(Yp,N_He):
    x=(1-Yp*(1-N_He/4))/(1-Yp/2)
    return x

def density(z,rho_g0):
    rho_g=rho_g0*(1+z)**3
    return rho_g

def ionized_electron_number(z):
    n=ionized_elec(0.24,1)*density(z,rho_g0)/(mu_e*m_p)
    return n

def tau(z):
    tau=np.array([])
    for i in z:
        z_temp=np.linspace(0,i,n_points)
        tau_der=sigma_T*cc.c_light_Mpc_s*ionized_electron_number(z_temp)/((1+z_temp)*uf.H(z_temp))
        tau=np.append(tau,sp.integrate.trapz(tau_der,z_temp))
    return tau
#print (tau(z)[11])
#plt.plot(z,tau(z))
#plt.show()
'''
x=np.linspace(0,5,5)
y=np.linspace(0,5,5)
xx=x[:,np.newaxis]
yy=y[np.newaxis,:]
integrand1=xx*yy
print (integrand1)
f1=sp.integrate.trapz(integrand1,xx,axis=0)
print (f1)
integral=sp.integrate.trapz(f1,y,axis=0)
print (integral)

#k=np.linspace(k_min,k_max,n_points)
k=1.
ell=1.
k_p=np.linspace(k_min,k_max,n_points)
zz=z[:,np.newaxis]
kk=k_p[np.newaxis,:]
#const=8*np.pi**2/((cc.c_light_Mpc_s)*(2*ell+1)**3)*(sigma_T*rho_g0/(mu_e*m_p))**2
mu=k**2+kk**2-(k-kk)**2/(2*k*kk)
I=k*(k-2*kk*mu)*(1-mu**2)/(kk**2*(k**2+kk**2-2*k*kk*mu))
delta_sq=k**3/(2*np.pi**2)*uf.f(zz)**2*kk**2/(2*np.pi)**3*Mps_interpf(kk)*Mps_interpf(k-kk)*I
integrand=[(1+zz)**2*ionized_elec(0.24,1)**2*delta_sq*uf.chi(zz)*uf.H(zz)]
integrand=np.resize(integrand,(5,5))
print (integrand)
f=sp.integrate.trapz(integrand,zz,axis=0)
print (f)
C_l=sp.integrate.trapz(f,k_p,axis=0)
print (C_l)
'''

kp=np.linspace(k_min,k_max,n_points)
kp_norm=np.linalg.norm(kp)

def Delta_b(k,z):
    Delta_b=np.array([]) 
    for i in k:
        #mu=kk/i
        #mu=(i**2+kk**2-(np.abs(i-kk))**2)/(2*i*kk)
        #print (mu)
        #I=i**2/(kk**2*(i**2+kk**2))
        k_norm=np.linalg.norm(i)
        mu=np.dot(i,kp)/(k_norm*kp_norm)
        #print (mu)
        I=i*(i-2*kp*mu)*(1-mu**2)/(i**2+kp**2-2*i*kp*mu)
        constants=i**3*uf.f(z)**2/((2*np.pi**2)*(2*np.pi)**3)
        delta_sq=constants*Mps_interpf(kp)*Mps_interpf(np.abs(i-kp))*I
        Integral=sp.integrate.trapz(delta_sq,kp)
        Delta_b=np.append(Delta_b,Integral)
        Delta_b_sqrt=np.sqrt(Delta_b)
    return Delta_b_sqrt

k=np.linspace(1e-2,10,n_points)
plt.loglog(k,Delta_b(k,z=0)*k)
plt.ylim(1e-13,1e3)
plt.xlim(1e-2,10)
plt.show()
   
def C_l(ell):
    C_l=np.array([])
    for i in ell: 
        const=8*np.pi**2*T_rad**2/((cc.c_light_Mpc_s)*(2*i+1)**3)*(sigma_T*rho_g0/(mu_e*m_p))**2
        Delta_bsq=Delta_b(k,z)**2
        integrand=[const*(1+z)**2*ionized_elec(0.24,1)**2*Delta_bsq*uf.chi(z)*uf.H(z)]
        #print (np.shape(integrand))
        #integrand=np.resize(integrand,(n_points,n_points))
        #print (np.shape(integrand))
        C_l=np.append(C_l,sp.integrate.trapz(integrand,z))
    return C_l
ell=np.linspace(2,4000,n_points)
plt.loglog(ell,C_l(ell))
#plt.loglog(ell,ell*(ell+1)*C_l(ell)*1e12/(2*np.pi))
plt.show()
 