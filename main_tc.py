import sys
import numpy as np

from trajectory import SingleExcitationWithCollectiveCoupling

plotResult = False
printOutput = False
if '--print' in sys.argv:
    printOutput = True
if '--plot' in sys.argv: 
    plotResult=True
    from matplotlib import pyplot as plt
    #plt.style.use('classic')
    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman', size='10')


if 'param.in' in sys.argv:
    execfile('param.in')
else:
    dt = 0.001
    Ntimes = 1000
    Nskip = 100

    Nmol = 601
    Wmol =  0.0
    Wgrd = -1.0

    Wcav = 0.0
    Vcav = 0.1

    Nrad = 1001
    Wmax = 1.0
    damp = 0.005
    Vrad = 0.001/5

    Delta = 1.0
    TauC = 0.1

    hbar = 63.508
    mass = 10000.0  # Joul/mol(ps/A)^2
    Kconst = 145000.0  # Joul/mol/A^2
    staticCoup = 56.506266927 # 1/ps
    dynamicCoup = 187.412451975 # 1/ps/A
    kBT = 1245.0 #Joul/mol

model1 = SingleExcitationWithCollectiveCoupling(Nmol,Nrad)
# model2 = SingleExcitationWithCollectiveCoupling(Nmol,Nrad)

model1.initialHamiltonian_Radiation(Wgrd,Wmol,Vrad,Wmax,damp,useQmatrix=True)
# model2.initialHamiltonian_Radiation(Wgrd,Wmol,Vrad,Wmax,damp,useQmatrix=False)
# model.initialHamiltonian_Cavity(Wgrd,Wcav,Wmol,Vcav,Vrad,Wmax,damp,useQmatrix=True)

model1.initialXjVj_Gaussian(kBT,mass,Kconst)
model1.updateNeighborHarmonicOscillator(staticCoup,dynamicCoup)

# model2.initialXjVj_Gaussian(kBT,mass,Kconst)
# model2.updateNeighborHarmonicOscillator(staticCoup,dynamicCoup)

# model1.initialCj_Bright()
# model1.initialCj_middle()
model1.initialCj_Boltzman(hbar,kBT,most_prob=True)
# model2.initialCj_Boltzman(hbar,kBT,most_prob=True)

times = []
Pmol1, Pmol2 = [], []
IPR1, IPR2 = [], []
Prad, Pdamp = [], []
Prad_tot, Pdamp_tot = [], []
Xj_list, Vj_list = [], []
distr_list = []
Displacement_list = []
E0 = model1.getEnergy()
Pdamp_int = np.zeros((Nrad,1))
for it in range(Ntimes):
    # model1.propagateXjVj_velocityVerlet(dt)
    # model2.propagateXjVj_velocityVerlet(dt)

    # model1.updateDiagonalDynamicDisorder(Delta,TauC,dt)
    # model2.updateDiagonalDynamicDisorder(Delta,TauC,dt)
    # model1.updateNeighborDynamicDisorder(Delta,TauC,dt)
    # model1.updateNeighborHarmonicOscillator(staticCoup,dynamicCoup)
    # model2.updateNeighborHarmonicOscillator(staticCoup,dynamicCoup)

    # model1.propagateCj_RK4(dt)
    # model2.propagateCj_RK4(dt)
    model1.propagateCj_dHdt(dt)
    # model2.propagateCj_dHdt(dt)
    
    
    # Pdamp_int = Pdamp_int + (2*damp*dt)*np.abs(model2.Cj[model2.Irad:])**2

    if it%Nskip==0:
        times.append( it*dt )
        Pmol1.append( model1.getPopulation_system()  )
        # Pmol2.append( model2.getPopulation_system() )

        IPR1.append( model1.getIPR() )
        # IPR2.append( model2.getIPR() )

        # Prad.append( np.abs(model2.Cj[model2.Irad:])**2)
        # Pdamp.append( Pdamp_int )
        # Prad_tot.append( model2.getPopulation_radiation() )
        # Pdamp_tot.append( np.sum(Pdamp_int) )

        Xj_list.append(model1.Xj)
        Vj_list.append(model1.Vj)
        distr_list.append(np.abs(model1.Cj[model1.Imol:model1.Imol+model1.Nmol])**2)
        Displacement_list.append(model1.getDisplacement())
        if printOutput:
            print("{t}\t{d}\t{dP}\t{dE}".format(t=it*dt,d=model1.getDisplacement(), 
                                                dP=model1.getPopulation_system(), 
                                                dE=model1.getEnergy()-E0 ))

if not plotResult:
    # write to output 
    fpop = open('Pmol.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fpop.write("{t}\t{Pmol}\n".format(t=times[it],Pmol=Pmol1[it]))

    fdis = open('Displacement.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fdis.write("{t}\t{Displacement}\n".format(t=times[it],Displacement=Displacement_list[it]))


if plotResult:
    Prad = np.array(Prad)
    Pdamp = np.array(Pdamp)
    Xj_list, Vj_list = np.array(Xj_list).T, np.array(Vj_list).T
    distr_list = np.array(distr_list)

    fig, ax= plt.subplots(1,4, figsize=(16.0,3.0))
    ax[0].plot(times,Pmol1, '-r', lw=2, label='Q matrix', alpha=0.7)
    # ax[0].plot(times,Pmol2, '-k', lw=2, label='Explicit', alpha=0.7)
    ax[0].plot(times,np.exp(-Nmol*np.real(model1.Gamma)*np.array(times)), '--k', lw=2, alpha=0.7)
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$P_{mol}$")
    ax[0].legend()

    # Xgrid, Ygrid = np.meshgrid(model2.Erad, times)
    # Zgrid = Prad[:,:,0] + Pdamp[:,:,0]
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

    ax[3].plot(times,IPR1,lw=2, label='IPR1')
    # ax[3].plot(times,IPR2,lw=2, label='IPR2')
    ax[3].legend()


    fig2, ax2= plt.subplots(1,3, figsize=(16.0,3.0))
    ax2[0].plot(times,Displacement_list)
    for j in range(Nmol):
        ax2[1].plot(Xj_list[j],Vj_list[j])

    for it in range(len(times)):
        ax2[2].plot(distr_list[it])
    plt.show()  

