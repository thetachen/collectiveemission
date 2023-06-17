import numpy as np
from matplotlib import pyplot as plt

from trajectory import SingleExcitationWithCollectiveCoupling

dt = 0.001
Ntimes = 500000
Nskip = 20

Nmol = 40
Wmol = 0.0
Wgrd = -3.0#[-1.2,1.2]
Vndd = 0.0
Gamma_loc = 0.05

Nrad = 1001
Wmax = 1.0
damp = 0.005
Vrad = 0.001

Delta = 0.1
TauC = 0.1

Wcav = 0.0
Vcav = 0.5/np.sqrt(Nmol)
Gamma_cav = 0.05

DriveParam={
    'DriveType':    'ContinuousCos',
    'DriveAmplitude':   0.0005,   
    # 'DriveFrequency':   2.0,
    'DrivePulseCenter': 0.0,
    'DrivePulseWidth':  0.0,
}

fig, ax= plt.subplots(1,4, figsize=(16.0,3.0))

Jcav_list = []
Wdrv_list = -Wgrd + np.arange(-1.2,1.2,0.1)
for Wdrv in Wdrv_list:
    DriveParam['DriveFrequency'] = Wdrv
    model1 = SingleExcitationWithCollectiveCoupling(Nmol,Nrad)

    model1.initialHamiltonian_LossyCavity(Wgrd,Wcav,Wmol,Vndd,Vcav,Gamma_cav,Gamma_loc)
    # model1.updateDiagonalStaticDisorder(Delta)
    model1.initialCj_Ground()

   

    times = []
    Pmol1, Pmol2 = [], []
    Pgrd1, Pgrd2 = [], []
    IPR1, IPR2 = [], []
    Pcav1 = []
    ExternalDrive = []
    for it in range(Ntimes):
        # model1.updateDiagonalDynamicDisorder(Delta,TauC,dt)
        drive = model1.updateExternalDriving(DriveParam,it*dt)
        
        model1.propagateCj_RK4(dt)
        # model2.propagateCj_RK4(dt)
        # model1.propagateCj_dHdt(dt)
        model1.reset_Cgrd()

        if it%Nskip==0:
            times.append( it*dt )
            Pmol1.append( model1.getPopulation_system() )
            Pcav1.append( model1.getPopulation_cavity() )
            Pgrd1.append( model1.getPopulation_ground() )
            IPR1.append( model1.getIPR() )

            # print("{t}\t{Pcav}".format(t=it*dt,Pcav=Pcav1[-1]) )
    Jcav = model1.getPopulation_cavity()*Gamma_cav
    print(Wdrv,round(Wdrv+Wgrd,2),Jcav)
    Jcav_list.append(Jcav)

    if round(Wdrv+Wgrd,2) in [-0.5,0.0,0.5]:
        ax[0].plot(times,Pmol1,'-r', lw=2, label='Pmol', alpha=0.7)
        ax[0].plot(times,Pcav1,'-g', lw=2, label='Pcav', alpha=0.7)
        ax[2].plot(times,Pgrd1,'-b', lw=2, label='Pgrd', alpha=0.7)
        ax[0].set_xlabel("$t$")
        ax[0].set_ylabel("$P$")
        ax[0].legend()

ax[1].plot(Wdrv_list+Wgrd,Jcav_list,'-o')
# Xgrid, Ygrid = np.meshgrid(model2.Erad, times)
# Zgrid = Prad[:,:,0] #+ Pdamp[:,:,0]
# ax[1].contourf(Xgrid, Ygrid, Zgrid, 100, cmap=plt.cm.jet)
# ax[1].set_xlabel(r"$\omega_\alpha$")
# ax[1].set_ylabel("$t$")

# for it in range(int(Ntimes/Nskip)-1):
#     ax[2].plot(model2.Erad,Pdamp[it]+Prad[it],label=str((it+1)*Nskip*dt))
# # ax[2].plot(model.Erad,Pdamp[-1]+Prad[-1],label=str((it+1)*Nskip*dt))
# if hasattr(model2, 'Icav'):
#     ax[2].axvline(x=np.sqrt(Nmol)*Vcav)
#     ax[2].axvline(x=-np.sqrt(Nmol)*Vcav)
# ax[2].legend()

# ax[3].plot(model2.Erad,Pdamp[-1]+Prad[-1])
# ax[3].legend()


# fig2, ax2= plt.subplots(1,3, figsize=(16.0,3.0))
# for j in range(Nmol):
#     ax2[1].plot(Xj_list[j],Vj_list[j])

# for it in range(len(times)):
#     ax2[2].plot(distr_list[it])
plt.show()

