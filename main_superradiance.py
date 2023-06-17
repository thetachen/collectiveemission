import numpy as np
from matplotlib import pyplot as plt

from trajectory import SingleExcitationWithCollectiveCoupling

dt = 0.05
Ntimes = 4000
Nskip = 20

Nmol = 10
Wmol =  0.0
Wgrd = -1.0
Vndd = 0.0

Nrad = 1001
Wmax = 1.0
damp = 0.005
Vrad = 0.001

Delta = 0.1
TauC = 0.1
DriveParam={
    'DriveType':    'Pulse',
    'DriveAmplitude':   0.005,   
    'DriveFrequency':   1.3,
    'DrivePulseCenter': 100.0,
    'DrivePulseWidth':  25.0,
}

model1 = SingleExcitationWithCollectiveCoupling(Nmol,Nrad)
model2 = SingleExcitationWithCollectiveCoupling(Nmol,Nrad)

model1.initialHamiltonian_Radiation(Wgrd,Wmol,Vndd,Vrad,Wmax,damp,useQmatrix=True)
model2.initialHamiltonian_Radiation(Wgrd,Wmol,Vndd,Vrad,Wmax,damp,useQmatrix=False)
# model1.updateDiagonalStaticDisorder(Delta)
# model2.updateDiagonalStaticDisorder(Delta)

# model1.initialCj_Bright()
# model2.initialCj_Bright()

model1.initialCj_Ground()
model2.initialCj_Ground()


times = []
Pmol1, Pmol2 = [], []
IPR1, IPR2 = [], []
Prad, Pdamp = [], []
Prad_tot, Pdamp_tot = [], []
Pdamp_int = np.zeros((Nrad,1))
ExternalDrive = []
for it in range(Ntimes):
    # model1.updateDiagonalDynamicDisorder(Delta,TauC,dt)
    # model2.updateDiagonalDynamicDisorder(Delta,TauC,dt)
    drive = model1.updateExternalDriving(DriveParam,it*dt)
    drive = model2.updateExternalDriving(DriveParam,it*dt)
    
    # model1.propagateCj_RK4(dt)
    # model2.propagateCj_RK4(dt)
    model1.propagateCj_dHdt(dt)
    model2.propagateCj_dHdt(dt)
    
    Pdamp_int = Pdamp_int + (2*damp*dt)*np.abs(model2.Cj[model2.Irad:])**2

    if it%Nskip==0:
        times.append( it*dt )
        Pmol1.append( model1.getPopulation_system() )
        Pmol2.append( model2.getPopulation_system() )

        IPR1.append( model1.getIPR() )
        IPR2.append( model2.getIPR() )

        Prad.append( np.abs(model2.Cj[model2.Irad:])**2)
        Pdamp.append( Pdamp_int )

        Prad_tot.append( model2.getPopulation_radiation() )
        Pdamp_tot.append( np.sum(Pdamp_int) )
        ExternalDrive.append(drive)

        print("{t}\t{dP}".format(t=it*dt,dP=Pmol2[-1]+Prad_tot[-1]+Pdamp_tot[-1]) )

Prad = np.array(Prad)
Pdamp = np.array(Pdamp)

fig, ax= plt.subplots(1,4, figsize=(16.0,3.0))
ax[0].plot(times,Pmol1, '-r', lw=2, label='Q matrix', alpha=0.7)
ax[0].plot(times,Pmol2, '-k', lw=2, label='Explicit', alpha=0.7)
ax[0].plot(times,Prad_tot, '-', lw=2, label='radition', alpha=0.7)
ax[0].plot(times,Pdamp_tot, '-', lw=2, label='damped', alpha=0.7)
ax[0].plot(times,np.exp(-Nmol*np.real(model2.Gamma)*np.array(times)), '--k', lw=2, alpha=0.7)
ax[0].plot(times,np.array(ExternalDrive)/DriveParam['DriveAmplitude'],'-')
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$P_{mol}$")
ax[0].legend()

Xgrid, Ygrid = np.meshgrid(model2.Erad, times)
Zgrid = Prad[:,:,0] #+ Pdamp[:,:,0]
ax[1].contourf(Xgrid, Ygrid, Zgrid, 100, cmap=plt.cm.jet)
ax[1].set_xlabel(r"$\omega_\alpha$")
ax[1].set_ylabel("$t$")

for it in range(int(Ntimes/Nskip)-1):
    ax[2].plot(model2.Erad,Pdamp[it]+Prad[it],label=str((it+1)*Nskip*dt))
# ax[2].plot(model.Erad,Pdamp[-1]+Prad[-1],label=str((it+1)*Nskip*dt))
if hasattr(model2, 'Icav'):
    ax[2].axvline(x=np.sqrt(Nmol)*Vcav)
    ax[2].axvline(x=-np.sqrt(Nmol)*Vcav)
ax[2].legend()

ax[3].plot(model2.Erad,Pdamp[-1]+Prad[-1])
ax[3].legend()


# fig2, ax2= plt.subplots(1,3, figsize=(16.0,3.0))
# for j in range(Nmol):
#     ax2[1].plot(Xj_list[j],Vj_list[j])

# for it in range(len(times)):
#     ax2[2].plot(distr_list[it])
plt.show()

