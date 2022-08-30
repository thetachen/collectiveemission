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
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='Times New Roman', size='10')


if 'param.in' in sys.argv:
    exec(open('param.in').read())
else:
    dt = 0.001
    Ntimes = 5000 * 4
    Nskip = 100

    Nmol = 101
    Wmol =  0.0
    Wgrd = -1.0
    Vndd = 0.5

    Wcav = 0.0
    Vcav = 0.0

    useStaticDisorder = False
    useDynamicDisorder = True
    Delta = 1.0
    TauC = 0.1

    # hbar = 63.508
    # mass = 10000.0  # Joul/mol(ps/A)^2
    # Kconst = 145000.0  # Joul/mol/A^2
    # staticCoup = 0.0 # 1/ps
    # dynamicCoup = 0.0 # 1/ps/A
    # kBT = 1245.0 #Joul/mol
    
model1 = SingleExcitationWithCollectiveCoupling(Nmol,0)

model1.initialHamiltonian_BareCavity(Wgrd,Wcav,Wmol,Vndd,Vcav,Gamma=0.00)

# model1.initialCj_Polariton(-1) # initialize before disorder

# model1.initialXjVj_Gaussian(kBT,mass,Kconst)
# model1.updateNeighborHarmonicOscillator(staticCoup,dynamicCoup)
if useStaticDisorder:
    model1.updateNeighborStaticDisorder(Delta)
if useDynamicDisorder:
    model1.updateNeighborDynamicDisorder(Delta,TauC,dt)

# model1.initialCj_Bright()
model1.initialCj_middle()
# model1.initialCj_Cavity()
# model1.initialCj_Boltzman(hbar,kBT,most_prob=True)
# model1.initialCj_Polariton()

times = []
Pmol1, Pmol2 = [], []
IPR1, IPR2 = [], []

Xj_list, Vj_list = [], []
distr_list = []
Displacement_list = []
# E0 = model1.getEnergy()
for it in range(Ntimes):
    # model1.propagateXjVj_velocityVerlet(dt)
    # model2.propagateXjVj_velocityVerlet(dt)
    if useDynamicDisorder:
        # model1.updateDiagonalDynamicDisorder(Delta,TauC,dt)
        model1.updateNeighborDynamicDisorder(Delta,TauC,dt)
    # model1.updateNeighborHarmonicOscillator(staticCoup,dynamicCoup)

    model1.propagateCj_RK4(dt)
    # model1.dHdt = model1.dHdt*0.0
    # model1.propagateCj_dHdt(dt)
    
    if it%Nskip==0:
        times.append( it*dt )
        Pmol1.append( model1.getPopulation_system()  )
        # Pmol2.append( model2.getPopulation_system() )

        IPR1.append( model1.getIPR() )
        # IPR2.append( model2.getIPR() )


        # Xj_list.append(model1.Xj)
        # Vj_list.append(model1.Vj)
        distr = np.abs(model1.Cj[model1.Imol:model1.Imol+model1.Nmol])**2
        distr_list.append(distr[:,0])
        Displacement_list.append(model1.getDisplacement())
        if printOutput:
            print("{t}\t{d}\t{dP}".format(t=it*dt,d=np.sqrt(model1.getDisplacement()),dP=model1.getPopulation_system()))
                                                # dE=model1.getEnergy()-E0 ))

if not plotResult:
    # write to output 
    fpop = open('Pmol.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fpop.write("{t}\t{Pmol}\n".format(t=times[it],Pmol=Pmol1[it]))

    fdis = open('Displacement.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fdis.write("{t}\t{Displacement}\n".format(t=times[it],Displacement=Displacement_list[it]))

    fwfn = open('wfn.dat'+sys.argv[-1], 'w')
    for j in range(model1.Nmol):
        fwfn.write(str(j))
        for i in range(len(distr_list)):
            fwfn.write('\t'+'{:2.10f}'.format(distr_list[i][j][0]))
        fwfn.write('\n')

if plotResult:

    distr_list = np.array(distr_list)

    fig, ax= plt.subplots(1,4, figsize=(16.0,3.0))
    ax[0].plot(times,Pmol1, '-r', lw=2, label='Q matrix', alpha=0.7)
    # ax[0].plot(times,Pmol2, '-k', lw=2, label='Explicit', alpha=0.7)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("$P_{mol}$")
    ax[0].legend()

    # ax[1].plot(times,np.array(Displacement_list)/np.array(Pmol1)*Pmol1[0])
    ax[1].plot(times,np.sqrt(np.array(Displacement_list)))
    # for j in range(Nmol):
    #     ax2[1].plot(Xj_list[j],Vj_list[j])
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('Displacement')

    Xgrid = np.array(range(len(distr_list[0])))
    Ygrid = times
    Xgrid, Ygrid = np.meshgrid(Xgrid, Ygrid)
    Zgrid = np.log(distr_list)
    CS = ax[2].contourf(Xgrid, Ygrid, Zgrid, 100, cmap=plt.cm.jet)
    ax[2].set_xlabel('sites')
    ax[2].set_ylabel('time')
    plt.tight_layout()
    plt.show()  

