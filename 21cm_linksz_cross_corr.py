# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:55:01 2018

@author: KaderF
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import useful_functions as uf
import distance as cd
import density as den
import constants as cc
import reionization as cr
import perturbation as cp
import lin_ksz_method2 as lksz
import twentyonecm_autocorr as tcm
import matplotlib.patches as mpatches


z_r=lksz.z_r
chi_r=lksz.chi_r
T_rad=2.73*1e6
def k(i,y,z):
    k=np.sqrt(uf.kpar(y,z)**2+(i/uf.chi(z))**2)
    return k

def mu_k(i,y,z):
    mu_k=uf.kpar(y,z)/k(i,y,z)
    return mu_k

def Cl_21_lksz(ell,y,z):
    #Cl=[]
    Cl=np.array([])

    for i in ell:
        a=uf.b_HI+uf.f(z)*mu_k(i,y,z)**2
        #print (a)
        delta_tcm=(uf.T_mean(z)*a*uf.D_1(z))/(uf.chi(z)**2*uf.r(z))
        #print (delta_tcm)
        delta_lksz=T_rad*lksz.f*uf.H0/cc.c_light_Mpc_s*lksz.g(chi_r)*np.abs(np.sin(uf.kpar(y,z_r)*chi_r))*mu_k(i,y,z_r)/k(i,y,z_r)*lksz.Mps_interpf(k(i,y,z_r))
        #print (delta_lksz)
        Cl_one=delta_tcm*delta_lksz
        Cl=np.append(Cl,Cl_one)
    return Cl  
'''
y=np.linspace(0,100,100)
def sin(y):
    sin=np.abs(np.sin(uf.kpar(y,z_r)*uf.chi(z_r)))
    return sin
plt.plot(y,sin(y))
plt.show()
'''
ell=np.linspace(0,1000,1000)
#plt.loglog(ell,Cl_21_lksz(ell,y=5,z=z_r),'r')
#plt.loglog(ell,Cl_21_lksz(ell,y=1,z=z_r),'b')
#plt.loglog(ell,Cl_21_lksz(ell,y=2,z=z_r),'r')
plt.loglog(ell,Cl_21_lksz(ell,y=277,z=1),'r')
plt.loglog(ell,Cl_21_lksz(ell,y=750,z=1),'g')
plt.loglog(ell,Cl_21_lksz(ell,y=2032,z=1),'b')

red_patch = mpatches.Patch(color='red', label='y=277')
#green_patch = mpatches.Patch(color='red', label=r'$\rm{z=z_r}$')
green_patch = mpatches.Patch(color='green', label='y=750')
blue_patch=mpatches.Patch(color='blue',label='y=2032')
plt.legend(handles=[red_patch,green_patch,blue_patch])
plt.ylabel(r'$\rm{C_l^{21-p}(l) [\mu K^2]}$')
#plt.ylabel(r'g (chi) ($\rm{Mpc}^{-1}$)')

plt.xlabel('l')
plt.show()
