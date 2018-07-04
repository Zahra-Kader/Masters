# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mcint
import random
import numpy as np


matterpower = np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float)

lowlim=0.1
uplim=10.0
def integrand_29(x):
    chi     = x[0]
    chi2 =  x[1]
    k = x[2]
    l=1.0
    f1=chi**(-2)*np.exp(1j*k*(chi-chi2))*k**2/((k**2+(l**2/chi**2))**2)
    return f1

def integrand_30(x):
    chi     = x[0]
    chi2 =  x[1]
    k = x[2]
    f2=np.exp(1j*k*(chi-chi2))/k**2
    return f2

def sampler():
    while True:
        chi     = random.uniform(lowlim,uplim)
        chi2 = random.uniform(lowlim,uplim)
        k = random.uniform(lowlim,uplim)
        yield (chi, chi2, k)


domainsize = (uplim-lowlim)**3
print (domainsize)
#multiply together the difference between the limits of integration for all the integrals
#expected = 16*math.pow(math.pi,5)/3.

def I(input):
    for nmc in [1000, 10000]:
        random.seed(1)
        result, error = mcint.integrate(input, sampler(), measure=domainsize, n=nmc)
        #diff = abs(result - expected)
        print ("Using n = ", nmc)
        print ("Result = ", result, "estimated error = ", error)
        #print ("Known result = ", expected, " error = ", diff, " = ", 100.*diff/expected, "%")
        print (" ")

I1=I(integrand_29)
I2=I(integrand_30)

        
