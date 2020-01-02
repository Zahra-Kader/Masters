import numpy,corrcal2,time
from matplotlib import pyplot as plt
import corrcal2

f=open('signal_sparse2_test.dat')
n=numpy.fromfile(f,'int32',1);
isinv=(numpy.fromfile(f,'int32',1)[0]!=0);
nsrc=numpy.fromfile(f,'int32',1);
nblock=numpy.fromfile(f,'int32',1);
nvec=numpy.fromfile(f,'int32',1);
lims=numpy.fromfile(f,'int32',(nblock+1))
diag=numpy.fromfile(f,'float64',n)
vecs=numpy.fromfile(f,'float64',nvec*n)
src=numpy.fromfile(f,'float64',nsrc*n)

n=n[0]
nsrc=nsrc[0]
nvec=nvec[0]

vecs=vecs.reshape([nvec,n])
if nsrc>0:
    src=src.reshape([nsrc,n])
mat=corrcal2.sparse_2level(diag,vecs,src,lims,isinv)
f=open('ant1.dat');ant1=numpy.fromfile(f,'int64')-1;f.close()
f=open('ant2.dat');ant2=numpy.fromfile(f,'int64')-1;f.close()
f=open('gtmp.dat');gvec=numpy.fromfile(f,'float64');f.close()
f=open('vis.dat');data=numpy.fromfile(f,'float64');f.close()



#scipy nonlinear conjugate gradient seems to work pretty well.
#note that it can use overly large step sizes in trials causing
#matrices to go non-positive definite.  If you rescale the gains
#by some large factor, this seems to go away.  If you routinely
#hit non-positive definite conditions, try increasing fac (or writing your
#own minimizer...)
from scipy.optimize import fmin_cg
niter=1000;
fac=1.;
asdf=fmin_cg(corrcal2.get_chisq,gvec*fac,corrcal2.get_gradient,(data,mat,ant1,ant2,fac))
t2=time.time()
fit_gains=asdf/fac

print (fit_gains,'fit gains')
print (gvec,'guess for gains')
print (fit_gains-gvec,'diff')
