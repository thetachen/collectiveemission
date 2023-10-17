import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt

from models import SingleExcitationWithCollectiveCoupling

parser = argparse.ArgumentParser()
parser.add_argument("-p","--plot",action="store_true")
parser.add_argument("-i","--input",type=str)
parser.add_argument("-v","--verbosity",action="store_true")
parser.add_argument("-s", "--save",type=int)
args = parser.parse_args()


if args.plot:
    from matplotlib import pyplot as plt

# if 'param.in' in sys.argv:
#     exec(open('param.in').read())
if  not args.input==None:
    exec(open(args.input).read())
else:
    dt = 0.001*5
    Ntimes = int(400000/5)
    Nskip = 10

    Nmol = 40
    Wmol = 0.0
    Wgrd = -2.0
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
    DisorderParam={
        # 'Type':     'None',
        # 'Type':     'Static_Diagonal',
        # 'Type':     'Gaussian_Static_Coupling',
        # 'Type':     'Uniform_Static_Coupling',
        # 'Type':     'Gaussian_Dynamic_Coupling',
        'Type':     'Uniform_Dynamic_Coupling',
        'Delta':            0.4*np.pi,
        'Tau':              1000.0,
    }

    DriveParam={
        'DriveType':        'ContinuousCos',
        'DriveAmplitude':   0.0005,   
        'DriveFrequency':   -Wgrd + 0.0,
        'DrivePulseCenter': 0.0,
        'DrivePulseWidth':  0.0,
    }



Jcav_list = []
Kuramoto_list = []
if True:
    model1 = SingleExcitationWithCollectiveCoupling(Nmol,Nrad)

    model1.initialHamiltonian_LossyCavity(Wgrd,Wcav,Wmol,Vndd,Vcav,Gamma_cav,Gamma_loc)
    if DisorderParam['Type'] == 'Static_Diagonal':
        model1.updateDiagonalStaticDisorder(DisorderParam['Delta'])
    elif DisorderParam['Type'] == 'Gaussian_Static_Coupling':
        model1.updateCouplingStaticDisorder(DisorderParam['Delta'],Vcav)
    elif DisorderParam['Type'] == 'Uniform_Static_Coupling':
        model1.updateCouplingStaticDisorder(DisorderParam['Delta'],Vcav,uniform_dist=True)
    elif DisorderParam['Type'] == 'Dyanmic_Diagonal':
        model1.updateDiagonalDynamicDisorder(DisorderParam['Delta'],DisorderParam['Tau'],dt)
    elif DisorderParam['Type'] == 'Gaussian_Dynamic_Coupling':
        model1.updateCouplingDynamicDisorder(DisorderParam['Delta'],DisorderParam['Tau'],Vcav,dt)
    elif DisorderParam['Type'] == 'Uniform_Dynamic_Coupling':
        model1.updateCouplingDynamicDisorder(DisorderParam['Delta'],DisorderParam['Tau'],Vcav,dt,uniform_dist=True)
    # model1.initialCj_Ground()
    model1.initialCj_Random()
   

    times = []
    Pmol1, Pmol2 = [], []
    Pgrd1, Pgrd2 = [], []
    IPR1, IPR2 = [], []
    Pcav1 = []
    ExternalDrive = []
    Order1, Phase1 = [], [] 
    for it in range(Ntimes):
        if DisorderParam['Type'] == 'Dyanmic_Diagonal':
            model1.updateDiagonalDynamicDisorder(DisorderParam['Delta'],DisorderParam['Tau'],dt)
        elif DisorderParam['Type'] == 'Gaussian_Dynamic_Coupling':
            model1.updateCouplingDynamicDisorder(DisorderParam['Delta'],DisorderParam['Tau'],Vcav,dt)
        elif DisorderParam['Type'] == 'Uniform_Dynamic_Coupling':
            model1.updateCouplingDynamicDisorder(DisorderParam['Delta'],DisorderParam['Tau'],Vcav,dt,uniform_dist=True)
        drive = model1.updateExternalDriving(DriveParam,it*dt,dt)
        
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
            order, phase = model1.getOrderParameter()
            Order1.append(order)
            Phase1.append(phase)
            # print("{t}\t{Pcav}".format(t=it*dt,Pcav=Pcav1[-1]) )

    period = 2*np.pi/np.abs(DriveParam['DriveFrequency'])
    I_period = np.argmin(np.abs(times-period))

    Jcav_list.append(np.average(Pcav1[-I_period:])*Gamma_cav)
    Kuramoto_list.append(np.average(Order1[-I_period:]))
    
    fpop = open('Population.dat'+str(args.save), 'w') 
    for it in range(len(times)):
        if it%Nskip==0:
            fpop.write("{t}\t{Pmol}\t{Pcav}\t{Pgrd}\n".format(t=times[it],Pmol=Pmol1[it],Pcav=Pcav1[it],Pgrd=Pgrd1[it])) 

    forder = open('Kuramoto.dat'+str(args.save), 'w') 
    for it in range(len(times)):
        if it%Nskip==0:
            forder.write("{t}\t{Order}\t{Phase}\t{IPR}\n".format(t=times[it],Order=float(Order1[it]),Phase=float(Phase1[it]),IPR=float(IPR1[it])))    


    if args.plot:
        fig, ax= plt.subplots(1,4, figsize=(16.0,3.0))
        ax[0].plot(times,Pmol1,'-r', lw=2, label='Pmol')
        ax[0].set_xlabel("$t$")
        ax[0].set_ylabel("$Pmol$")
        ax[1].plot(times,Pcav1,'-g', lw=2, label='Pcav')
        ax[1].set_xlabel("$t$")
        ax[1].set_ylabel("$Pcav$")

        ax[2].plot(times,Order1,'-', lw=2, label='Order')
        ax[2].set_xlabel("$t$")
        ax[2].set_ylabel("Order")
        # ax[3].plot(times,Phase1,'-', lw=2, label='Phase, w='+str(round(Wdrv+Wgrd,2)), alpha=0.7)
        ax[2].legend()

        ax[3].plot(times,IPR1,'-', lw=2, label='Order')
        ax[3].set_xlabel("$t$")
        ax[3].set_ylabel("IPR")
        ax[3].legend()

plt.show()

