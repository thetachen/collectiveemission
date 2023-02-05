import sys
import numpy as np
from copy import deepcopy

from trajectory import SingleExcitationWithCollectiveCoupling


plotResult = False
SanityCheck = False
printOutput = False
if '--print' in sys.argv:
    printOutput = True
if '--test' in sys.argv:
    SanityCheck = True
if '--plot' in sys.argv: 
    plotResult=True
    from matplotlib import pyplot as plt
    #plt.style.use('classic')
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='Times New Roman', size='10')


if 'param.in' in sys.argv:
    exec(open('param.in').read())
else:
    dt = 0.01
    Ntimes = 20000
    Nskip = 10

    Nmol = 101
    Wmol =  0.0
    Wgrd = -1.0
    Vndd = -0.1

    Wcav = 0.0 + 2.0*Vndd
    Vcav = 0.0
    Kcav = 0  #Kcav*pi/Nmol

    useStaticNeighborDisorder = False
    useDynamicNeighborDisorder = False
    DeltaNN = 0.0
    TauNN = 0.0

    useStaticDiagonalDisorder = False
    useDynamicDiagonalDisorder = False
    DeltaDD = 0.0
    TauDD = 0.0

    kBT = 0.1
    # hbar = 63.508
    # mass = 10000.0  # Joul/mol(ps/A)^2
    # Kconst = 145000.0  # Joul/mol/A^2
    # staticCoup = 0.0 # 1/ps
    # dynamicCoup = 0.0 # 1/ps/A
    # kBT = 1245.0 #Joul/mol
    
model1 = SingleExcitationWithCollectiveCoupling(Nmol,0)
model1.initialHamiltonian_nonHermitian(Wgrd,Wcav,Wmol,Vndd,Vcav,Kcav,Gamma=0.0)

useNodisorder=False
if useStaticNeighborDisorder:
    model1.updateNeighborStaticDisorder(DeltaNN)
elif useDynamicNeighborDisorder:
    model1.updateNeighborDynamicDisorder(DeltaNN,TauNN,dt)
elif useStaticDiagonalDisorder:
    model1.updateDiagonalStaticDisorder(DeltaDD)
elif useDynamicDiagonalDisorder:
    model1.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)
else:
    useNodisorder=True

if useDynamicNeighborDisorder or useDynamicDiagonalDisorder or SanityCheck:
    print("Run time evolution for evaluation of CJJ")
    model2 = deepcopy(model1)
    model2.initialCall_Eigenstate(Vcav)
    model2.initialParition_Boltzmann(kBT)

    times = []
    Javg2_list = []
    CJJavg2_list = []

    for it in range(Ntimes):
        if useDynamicNeighborDisorder:
            model2.updateNeighborDynamicDisorder(DeltaNN,TauNN,dt)
        if useDynamicDiagonalDisorder:
            model2.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)

        Javg2, CJJavg2 = model2.getCurrentCorrelation_all()
        
        model2.propagateCall_RK4(dt)
        model2.propagateJ0Call_RK4(dt)
        
        if it%Nskip==0:
            times.append( it*dt )
            Javg2_list.append(Javg2)
            CJJavg2_list.append(CJJavg2)
    
    CJJavg_list = np.array(CJJavg2_list)
    EigEng_list = model2.Eall

if useStaticNeighborDisorder or useStaticDiagonalDisorder or useNodisorder or SanityCheck:
    print("Use the stationary evaluation of CJJ")
    model1.initialCj_Eigenstate(Vcav,initial_state=0)
    CJJavg1 = model1.getCurrentCorrelation_stationary(dt,Ntimes,kBT,Vcav)

    times = []
    Javg1_list = []
    CJJavg1_list = []
    for it in range(Ntimes):
        if it%Nskip==0:
            times.append( it*dt )
            CJJavg1_list.append(CJJavg1[it])

    CJJavg_list = np.array(CJJavg1_list)
    EigEng_list = model1.Eall

if SanityCheck:
    for it in range(len(times)):
        print(it, CJJavg1_list[it], CJJavg2_list[it], np.abs(CJJavg1_list[it]-CJJavg2_list[it]))
else:
    fcorr = open('CJJ_therm.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        if it%Nskip==0:
            fcorr.write("{t}\t{CJJ_real}\t{CJJ_imag}\n".format(t=times[it],CJJ_real=np.real(CJJavg_list)[it],CJJ_imag=np.imag(CJJavg_list)[it]))

    feigen = open('Eigen.dat'+sys.argv[-1], 'w')
    for i in range(len(EigEng_list)):
        feigen.write("{EigEng}\n".format(EigEng=EigEng_list[i]))    
exit()
