import numpy as np
from matplotlib import pyplot as plt

from trajectory import Trajectory_TCmodel

dt = 0.05
Ntimes = 2000
Nskip = 20

Nmol = 10 
Wmol =  0.0
Wgrd = -1.0

Wcav = 0.0
Vcav = 0.1

Nrad = 1001
Wmax = 1.0
damp = 0.005
Vrad = 0.001


traj = Trajectory_TCmodel(Nmol,Nrad)
# traj.initialHamiltonian_Free(Wgrd,Wmol,Vrad,Wmax,damp)
traj.initialHamiltonian_Cavity(Wgrd,Wcav,Wmol,Vcav,Vrad,Wmax,damp)
traj.initialState(InitialState="Bright")


times = []
Pmol1 = []
Pmol2 = []
Prad, Pdamp = [], []
Prad_tot, Pdamp_tot = [], []
Pdamp_int = np.zeros((Nrad,1))
for it in range(Ntimes):
    traj.propagateCj_RK4(dt)

    
    Pdamp_int = Pdamp_int + (2*damp*dt)*np.abs(traj.Cj2[traj.Irad:])**2

    if it%Nskip==0:
        print(it)
        times.append( it*dt )
        Pmol1.append( np.linalg.norm(traj.Cj1[traj.Imol:traj.Irad])**2 )
        Pmol2.append( np.linalg.norm(traj.Cj2[traj.Imol:traj.Irad])**2 )

        Prad.append( np.abs(traj.Cj2[traj.Irad:])**2)
        Pdamp.append( Pdamp_int )

        Prad_tot.append( np.linalg.norm(traj.Cj2[traj.Irad:])**2 )
        Pdamp_tot.append( np.sum(Pdamp_int) )
Prad = np.array(Prad)
Pdamp = np.array(Pdamp)
print(Prad.shape, Pdamp.shape)

fig, ax= plt.subplots(1,3, figsize=(16.0,3.0))
ax[0].plot(times,Pmol1, '-r', lw=2, label='Q matrix', alpha=0.7)
ax[0].plot(times,Pmol2, '-k', lw=2, label='Explicit', alpha=0.7)
ax[0].plot(times,np.exp(-Nmol*np.real(traj.Gamma)*np.array(times)), '--k', lw=2, alpha=0.7)
ax[0].legend()



Xgrid, Ygrid = np.meshgrid(traj.Erad, times)
Zgrid = Prad[:,:,0] + Pdamp[:,:,0]
CS = ax[1].contourf(Xgrid, Ygrid, Zgrid, 100, cmap=plt.cm.jet)


for it in range(int(Ntimes/Nskip)-1):
    ax[2].plot(traj.Erad,Pdamp[it]+Prad[it],label=str((it+1)*Nskip*dt))
# ax[2].plot(traj.Erad,Pdamp[-1]+Prad[-1],label=str((it+1)*Nskip*dt))
if hasattr(traj, 'Icav'):
    ax[2].axvline(x=np.sqrt(Nmol)*Vcav)
    ax[2].axvline(x=-np.sqrt(Nmol)*Vcav)
ax[2].legend()
plt.show()

