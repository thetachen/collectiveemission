#!/home/theta/anaconda2/bin/python
import sys
from os import mkdir
from shutil import copyfile
import numpy as np
from random import random
plotresult = False
printoutput = False
if '--print' in sys.argv:
    printoutput = True
if '--plot' in sys.argv: 
    plotresult=True
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MultipleLocator, AutoLocator, MaxNLocator
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    #plt.style.use('classic')
    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman', size='10')

#srcdir = '/home/theta/Qsync/sync/superadiance-disorder/'
#sys.path.append(srcdir)
#from funcs import *

def GenerateGaussianProcess(Delta,TauC,Ntimes,dt):
    # simulate Gaussian process
    # cf. George B. Rybicki's note
    # https://www.lanl.gov/DLDSTP/fast/OU_process.pdf
    if TauC==0.0:
        ri = 0.0
    else:
        ri = np.exp(-dt/TauC)

    Omega = np.zeros(Ntimes)
    Omega[0] = np.random.normal(0.0,Delta)

    for it in range(1,Ntimes):
        mean_it = ri*Omega[it-1]
        sigma_it = Delta*np.sqrt(1.0-ri**2)
        Omega[it] = np.random.normal(mean_it,sigma_it)
    return Omega

if 'param.in' in sys.argv:
    execfile('param.in')
else:
    param = {
        'dt':       0.05,
        'Ntimes':   10000,
        'Nsec':     100,
        'Nslice':   2000,
        'Nmol':     2,
        'Wmol':     0.0,
        'Wgrd':     -0.5,
        'Nrad':     601,
        'Wrad_max': 0.6,
        'damping':  0.005,
        'Vrad':     0.001,
        'Wrad_width':       0.4,
        'DriveAmplitude':   0.0,
        'DriveFrequency':   0.5,
        'DynamicalDisorder':    True,
        'StaticDisorder':       False,
        'Delta':    0.02,
        'TauC':     10.0,
    }

for key, value in param.items():
    globals()[key] = value


# for itraj in range(Ntraj):
if True:

    if StaticDisorder: 
        Ws = np.random.normal(Wmol,Delta,Nmol)
        #print(Ws)

    if DynamicalDisorder:
        Wt = np.zeros((Nmol,Ntimes))
        for i in range(Nmol):
            Wt[i] = GenerateGaussianProcess(Delta,TauC,Ntimes,dt)
        Wt = Wt.T
 
    # Construct the Hamiltonian
    Hgrd = np.eye(1) * Wgrd
    Hmol = np.eye(Nmol) * Wmol
    Erad = np.zeros(Nrad)
    Hrad = np.zeros((Nrad,Nrad),complex)
    Vmolrad = np.zeros((Nrad,Nmol),complex)
    Vmolgrd = np.zeros((Nmol,1),complex)
    Vgrdrad = np.zeros((Nrad,1),complex)

    # Construct the molecule-radiation coupling
    Gamma = 0.0
    for j in range(Nrad):
        Erad[j] = ( j - (Nrad-1)/2 ) * Wrad_max *2.0/(Nrad-1)
        Hrad[j,j] = Erad[j] - 1j*damping
        for i in range(Nmol):
            Vmolrad[j,i] = Vrad# * (Wrad_width**2) / ( Erad[j]**2 + Wrad_width**2 )
        Gamma += -2.0*1j*(Vmolrad[j,0]**2)/Hrad[j,j] # set to be the same: 0
    #Gamma = 1j*Gamma*(Vrad**2)
    Gamma = np.real(Gamma)
    print(Gamma)

    # Construct the molecule-ground coupling
    for imol in range(Nmol):
        Vmolgrd[imol,0] = 1.0

    # bright state is symmetric
    Cbright1 = np.ones((Nmol,1),complex)/np.sqrt(Nmol) 
    Cbright1 = np.vstack( (np.zeros((1,1),complex), Cbright1) )
    Cbright2 = np.vstack( (Cbright1,np.zeros((Nrad,1),complex)) )

    ### dark state is chosen as (1,-1,0,....,0)
    Cdark1 = np.zeros((Nmol,1),complex)
    if Nmol>1:
        Cdark1[0,0] = 1.0/np.sqrt(2.0)
        Cdark1[1,0] = -1.0/np.sqrt(2.0)
    Cdark1 = np.vstack( (np.zeros((1,1),complex), Cdark1) )
    Cdark2 = np.vstack( (Cdark1,np.zeros((Nrad,1),complex)) )
    
    # Initialize state vector
    Cvec1 = np.ones((Nmol,1),complex)/np.sqrt(Nmol)
    Cvec1 = np.vstack( (np.zeros((1,1),complex), Cvec1) )
    # Cvec1 = np.zeros((Nmol,1),complex)
    # Cvec1 = np.vstack( (np.ones((1,1),complex), Cvec1) )
    Cvec2 = np.vstack( (Cvec1,np.zeros((Nrad,1),complex)) )

    # Propagation
    #eigval, eigvec = np.linalg.eig(Hamiltonian)
    #Udt = np.dot(eigvec, np.dot( np.diag(np.exp(-1j*eigval*dt)), np.linalg.inv(eigvec)))    
    times = []
    Emol = []
    for imol in range(Nmol): Emol.append([])
    Pmol1 = []
    Pmol2 = []
    Prad = []
    Pbright1 = []
    Pbright2 = []
    Pdark1 = []
    Pdark2 = []
    for it in range(1,Ntimes):
        drive = DriveAmplitude*np.sin(DriveFrequency*it*dt)
        Ht = np.vstack(( np.hstack(( Hgrd,          Vmolgrd.T*drive,    Vgrdrad.T )),
                         np.hstack(( Vmolgrd*drive, Hmol,               Vmolrad.T )),
                         np.hstack(( Vgrdrad,       Vmolrad,            Hrad )) ))
        HQ = np.vstack(( np.hstack(( Hgrd,          Vmolgrd.T*drive )),
                         np.hstack(( Vmolgrd*drive, Hmol - 1j*(Gamma/2)*np.ones((Nmol,Nmol)) )) ))
        
        if StaticDisorder:
            for imol in range(Nmol):
                Ht[1+imol,1+imol] += Ws[imol]
                HQ[1+imol,1+imol] += Ws[imol]            
        if DynamicalDisorder:
            for imol in range(Nmol): 
                Ht[1+imol,1+imol] += Wt[it,imol]
                HQ[1+imol,1+imol] += Wt[it,imol]

        ### RK4 propagation 
        K1 = -1j*np.dot(HQ,Cvec1)
        K2 = -1j*np.dot(HQ,Cvec1+dt*K1/2)
        K3 = -1j*np.dot(HQ,Cvec1+dt*K2/2)
        K4 = -1j*np.dot(HQ,Cvec1+dt*K3)
        Cvec1 += (K1+2*K2+2*K3+K4)*dt/6

        K1 = -1j*np.dot(Ht,Cvec2)
        K2 = -1j*np.dot(Ht,Cvec2+dt*K1/2)
        K3 = -1j*np.dot(Ht,Cvec2+dt*K2/2)
        K4 = -1j*np.dot(Ht,Cvec2+dt*K3)
        Cvec2 += (K1+2*K2+2*K3+K4)*dt/6

        ### Output/
        if it%Nsec==0:
            times.append( it*dt )
            for imol in range(Nmol):
                Emol[imol].append( Ht[1+imol,1+imol] )
            Pmol1.append( np.linalg.norm(Cvec1[1:Nmol+1])**2 )
            Pmol2.append( np.linalg.norm(Cvec2[1:Nmol+1])**2 )
            Pbright1.append( np.abs(np.dot(Cbright1.T,Cvec1)[0,0])**2 )
            Pbright2.append( np.abs(np.dot(Cbright2.T,Cvec2)[0,0])**2 )
            Pdark1.append( np.abs(np.dot(Cdark1.T,Cvec1)[0,0])**2 )
            Pdark2.append( np.abs(np.dot(Cdark2.T,Cvec2)[0,0])**2 )
            if printoutput:
                print("{t}\t{P1}\t{P2}".format(t=times[-1],P1=Pmol1[-1],P2=Pmol2[-1]))
                print("\t{Pb1}\t{Pb2}".format(Pb1=Pbright1[-1],Pb2=Pbright2[-1]))
                print("\t{Pd1}\t{Pd2}".format(Pd1=Pdark1[-1],Pd2=Pdark2[-1]))
        if it%Nslice==0:
            Prad.append( np.abs(Cvec2[Nmol+1:])**2)

    fpop = open('pop.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fpop.write("{t}\t{P1}\t{P2}\t{Pb1}\t{Pb2}\t{Pd1}\t{Pd2}\n".format(t=times[it],
                    P1=Pmol1[it],P2=Pmol2[it],
                    Pb1=Pbright1[it],Pb2=Pbright2[it],
                    Pd1=Pdark1[it],Pd2=Pdark2[it]))
    frad = open('rad.dat'+sys.argv[-1], 'w')
    for j in range(len(Erad)):
        frad.write(str(round(Erad[j],4)))
        for i in range(len(Prad)):
            frad.write('\t'+'{:2.8f}'.format(Prad[i][j][0]))
        frad.write('\n')
    #sys.stdout = file
    fpop.close()
    frad.close()

if plotresult:
    times=np.array(times)

    fig, ax= plt.subplots(1,3, figsize=(16.0,3.0))
    #ax[0].plot(times,Pmol1, '-r', lw=2, label='Q matrix', alpha=0.7)
    ax[0].plot(times,Pmol2, '-k', lw=2, label='Explicit Radiation', alpha=0.7)
    #ax[0].plot(times,Pbright1, '--r', lw=2, label='Q matrix: Bright', alpha=0.7)
    ax[0].plot(times,Pbright2, '--r', lw=2, label='Explicit Radiation: Bright', alpha=0.7)
    #ax[0].plot(times,Pdark1, ':r', lw=2, label='Q matrix: Dark', alpha=0.7)
    ax[0].plot(times,Pdark2, ':b', lw=2, label='Explicit Radiation: Dark', alpha=0.7)
    ax[0].set_xlim([0,Ntimes*dt])
    ax[0].set_ylim([0,1])
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("population")
    ax[0].plot(times,np.exp(-Nmol*np.real(Gamma)*times), '--k', lw=2, alpha=0.7)
    ax[0].legend()

    for imol in range(Nmol):
        ax[1].plot(times,Emol[imol],label=str(imol))
    for it in range(Ntimes/Nslice-1):
        ax[2].plot(Erad,Prad[it],label=str((it+1)*Nslice*dt))
        ax[2].legend()
    
    plt.tight_layout()
    plt.show()



