import matplotlib.pyplot as plt
import hirax_tools as ht
from hirax_tools.utils import pointing
import numpy as np
import scipy as sp
from astropy import units
from astropy import coordinates as coords

rd = ht.RawData('/home/zahra/Downloads/01891349_0000.h5', transpose_on_init=True) #uncomment this when opening a new console


T_source=10**7 #This is a fake value at the moment
'''
r=[100,200,300,400,500,590,600,700,800]

#check autocorr plots for different frequencies and dishes
for i in r:
    band_ind=i
    filtered = rd.filtered_on_time((3, 3), filter_type='low', char_freqs=1/0.5)
    y=np.abs(filtered[:, band_ind])
    plt.plot(y)
    plt.show()



unix_times=rd.times
from astropy.time import Time
times = Time(unix_times, format='unix')

hartrao = coords.EarthLocation.from_geodetic(
    lon=27.6853931*units.deg,
    lat=-25.8897515*units.deg,
    height=1415.710*units.m,
    ellipsoid='WGS84')

alt, az = (90-17.1)*units.deg, 180*units.deg
altazs = coords.SkyCoord(frame='altaz', alt=alt*np.ones(len(times)),az=az*np.ones(len(times)),
                         location=hartrao, obstime=times)
j2000_pointing = altazs.transform_to(coords.ICRS)

fornax = coords.SkyCoord(ra='03h22m41.7s', dec='-37d12m30s')
cen=coords.SkyCoord(ra='13h25m27.6s', dec='-43d01m11s')
pointing_separation = j2000_pointing.separation(cen)

plt.plot(pointing_separation)
plt.show()
'''


x=pointing(rd.times, 79.8, 180, frame='Galactic').b
x=x.deg


def signal(band_ind):
    filtered = rd.filtered_on_time((3, 3), filter_type='low', char_freqs=1/0.5)
    y=np.abs(filtered[:, band_ind])
    return y


def Gauss(x, a, x0, sigma,const):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))+const

const=1.9e7
r=np.arange(0,1025,1)

Tsys=np.array([])
x_freq=np.array([])
for i in r:
    try:
        #print (i)
        y=signal(np.int(i))
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
        popt,pcov = sp.optimize.curve_fit(Gauss, x, y, p0=[max(y), mean, sigma,const])
        #plt.plot(x,y)
        #plt.plot(x, Gauss(x, *popt))
        #plt.show()
        pcov_diag=np.diagonal(pcov) #take sqrt
        #print (pcov_diag)
        for j in pcov_diag:
            if j>1e9:
                break
            elif j<0:
                break
            else:
                Y=max(Gauss(x, *popt))-popt[-1]
                if Y>0:
                    freq_MHz=rd.bands[np.int(i)]
                    x_freq=np.append(x_freq,freq_MHz)
                    T_sys_ind=T_source/(Y-1)
                    Tsys=np.append(Tsys,T_sys_ind)
    except:
        pass


plt.scatter(x_freq,Tsys)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Tsys')
plt.show()


# make a model that is a Gaussian + a constant:

'''
print (np.sqrt(np.diagonal(pcov)))
Y=max(Gauss(x,*popt))-const
print (Y)
T_sys=T_source/(Y-1)
print (T_sys,'Tsys')




plt.title('Signal amplitude as a function of galactic latitude for 527 MHz')
plt.plot(x,signal(band_ind))
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.xlabel('Galactic latitude (degrees)')
plt.ylabel('Signal Amplitude')
plt.show()

#print(rd.bands[band_ind])

x=np.array([527,566,605,644,682,722,761])
y=np.array([2.75,2.79,2.25,2.36,3.35,3.44,3.75])
plt.scatter(x,y)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Tsys (do not trust the values)')
'''
