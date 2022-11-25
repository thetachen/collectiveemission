import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

dt = 0.01
Ntimes = 5000


Nmol = 50
X = np.array(range(Nmol+1))*1.0
P = np.zeros(Nmol+1)
dPdt = np.zeros(Nmol+1)
Wplus = np.zeros((Nmol+1,Nmol+1))
Wminus = np.zeros((Nmol+1,Nmol+1))

W0 = 0.1
sigma = 0.05
Wrandom = np.random.normal(0.0,sigma,Nmol+1)

for n in range(Nmol):
    if n==0:
        Wplus[0,1] = Wrandom[0]
    else:
        Wplus[n,n+1] = Wplus[n-1,n] + Wrandom[n]
for n in range(Nmol):
    Wplus[n,n+1] = Wplus[n,n+1] + W0
    Wminus[n+1,n] = Wplus[n,n+1]
P[0]=1.0

times = []
displacement = []
fig, ax= plt.subplots(2,figsize=(4.0,3.0))
for it in range(Ntimes):
    times.append(it*dt)
    for n in range(Nmol):
        dPdt[n] = Wminus[n+1,n]*P[n+1] + Wplus[n-1,n]*P[n-1] - (Wplus[n,n+1]+Wminus[n,n-1])*P[n]
    P = P+ dt*dPdt
    displacement.append(np.sum(X**2*P))
    ax[0].plot(X,P)

times = np.array(times)
displacement = np.array(displacement)

cuts = [20.0,50.0]
cut1=np.argmin(np.abs(times-cuts[0]))
cut2=np.argmin(np.abs(times-cuts[1]))

res = stats.linregress(np.log10(times[cut1:cut2]), np.log10(displacement[cut1:cut2]))
slope = res.slope
intercept = res.intercept
print(slope)
ax[1].loglog(times,displacement)
ax[1].loglog(times[cut1:cut2],10.0**(slope*np.log10(times[cut1:cut2])+intercept),'--')
plt.show()