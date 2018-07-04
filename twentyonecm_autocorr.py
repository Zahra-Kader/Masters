# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:16:51 2018

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
import perturbation as cp
import lin_ksz_method2 as lksz
#rho_0=den.cosmo_densities(**cosmo)[1]*cc.M_sun_g #in units of g/Mpc^3

z=uf.z

def Mps_k(ell):
    #Cl=np.array([])
    Mps=np.array([])
    kpar=(2/uf.r(1))
    for i in ell:
        k=np.sqrt(kpar**2+(i/uf.chi(1))**2)
        Mps_one_ell=uf.Mps_interpf(k)
        #Mps_div_ell=Mps*kpar**4*(T_mean(1)*(kpar**2/k**2))**2/(chi(1)**2*r(1))
        #Cl=np.append(Cl,Mps_div_ell)
        Mps=np.append(Mps,Mps_one_ell)
    return Mps

kpar=(2./uf.r(1))
kperp=0.
Mps_zero_ell=uf.Mps_interpf(kpar)
#print (Mps_zero_ell)

ell=np.linspace(0,1000,1000)
##plt.loglog(ell,Mps_k(ell))
#plt.title('Mps div k**4')
##plt.show()

def Cl_21(ell,y,z):
    #Cl=[]
    Cl=np.array([])
    for i in ell:
        kpar=y/uf.r(z)
        k=np.sqrt(kpar**2+(i/uf.chi(z))**2)
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+uf.f(z)*mu_k_sq
        Cl_one=(uf.T_mean(z)*a*uf.D_1(z))**2*uf.Mps_interpf(k)/(uf.chi(z)**2*uf.r(z))
        Cl=np.append(Cl,Cl_one)
    return Cl
##plt.plot(z,uf.H(z))
#plt.ylabel('H(z)')
#plt.xlabel('z')
##plt.show()
##plt.loglog(1+z,uf.D_1(z))
#plt.xlim(10**3,10**0)
#plt.ylabel('Growth factor')
#plt.xlabel('1+z')
##plt.show()
##plt.loglog(ell,Cl_21(ell,y=277,z=1),'r')
##plt.loglog(ell,Cl_21(ell,y=750,z=1),'b')
##plt.loglog(ell,Cl_21(ell,y=2032,z=1),'g')
#plt.xlabel('l')
red_patch = mpatches.Patch(color='red', label='y=277')
blue_patch = mpatches.Patch(color='blue', label='y=750')
green_patch = mpatches.Patch(color='green', label='y=2032')
#plt.legend(handles=[red_patch,blue_patch,green_patch])
#plt.xlim(100,10**3)
###plt.show()


