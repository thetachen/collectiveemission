import sys
import numpy as np
from matplotlib import pyplot as plt

from trajectory import SingleExcitationWithCollectiveCoupling

if '--plot' in sys.argv: 
    plotResult=True
    from matplotlib import pyplot as plt

if 'param.in' in sys.argv:
    exec(open('param.in').read())
else:
    dt = 0.001
    Ntimes = 30000
    Nskip = 100
    
    Nmol = 40
    Wmol = 0.0
    Wgrd = -3.0
    Vndd = 0.0
    Gamma_loc = 0.0

    Nrad = 1001
    Wmax = 1.0
    damp = 0.005
    Vrad = 0.001

    Delta = 0.1
    TauC = 0.1

    Wcav = 0.0
    Vcav = 0.5/np.sqrt(Nmol)

    DriveParam={
        'DriveType':    'Pulse',
        'DriveAmplitude':   0.0005,   
        'DriveFrequency':   -Wgrd - Vcav*np.sqrt(Nmol), #+/-: upper and lower polariton
        'DrivePulseCenter': 100.0,
        'DrivePulseWidth':  25.0,
    }

    useStaticDiagonalDisorder = False
    useDynamicDiagonalDisorder = False
    DeltaDD = 0.0
    TauDD = 0.0

if True:
    
    model1 = SingleExcitationWithCollectiveCoupling(Nmol,Nrad)
    model2 = SingleExcitationWithCollectiveCoupling(Nmol,Nrad)

    
    model2.initialHamiltonian_LossyCavity_Radiation(Wgrd,Wcav,Wmol,Vndd,Vcav,Vrad,Wmax,damp,Gamma_loc)
    Gamma_cav = model2.Gamma_cav/2
    model1.initialHamiltonian_LossyCavity(Wgrd,Wcav,Wmol,Vndd,Vcav,Gamma_cav,Gamma_loc)
    if useStaticDiagonalDisorder:
        model2.updateDiagonalStaticDisorder(DeltaDD)
        model1.Ht = model2.Ht[:model2.Irad,:model2.Irad]
        # model1.updateDiagonalStaticDisorder(DeltaDD)
    
    model1.initialCj_Ground()
    model2.initialCj_Ground()
   

    times = []
    Pbrt1, Pdrk1 = [], []
    Pbrt2, Pdrk2 = [], []
    P_up1, P_lp1 = [], []
    P_up2, P_lp2 = [], []
    Pmol1, Pmol2 = [], []
    Pgrd1, Pgrd2 = [], []
    Prad, Pdamp = [], []
    Prad_tot, Pdamp_tot = [], []
    Pdamp_int = np.zeros((Nrad,1))
    IPR1, IPR2 = [], []
    Pcav1,Pcav2 = [], []
    ExternalDrive = []
    for it in range(Ntimes):
        if useDynamicDiagonalDisorder:
            model1.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)
            model2.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)
        drive = model1.updateExternalDriving(DriveParam,it*dt)
        drive = model2.updateExternalDriving(DriveParam,it*dt)

        model1.propagateCj_RK4(dt)
        model2.propagateCj_RK4(dt)
        # model1.propagateCj_dHdt(dt)
        # model1.reset_Cgrd()

        Pdamp_int = Pdamp_int + (2*damp*dt)*np.abs(model2.Cj[model2.Irad:])**2

        if it%Nskip==0:
            times.append( it*dt )
            Pmol1.append( model1.getPopulation_system() )
            Pmol2.append( model2.getPopulation_system() )
            Pcav1.append( model1.getPopulation_cavity() )
            Pcav2.append( model2.getPopulation_cavity() )
            Pgrd1.append( model1.getPopulation_ground() )
            Pbrt, Pdrk = model1.getPopulation_bright()
            P_up, P_lp = model1.getPopulation_polariton()
            Prad.append( np.abs(model2.Cj[model2.Irad:])**2)
            Pdamp.append( Pdamp_int )
            Prad_tot.append( np.linalg.norm(model2.Cj[model2.Irad:])**2 )
            Pdamp_tot.append( np.sum(Pdamp_int) )
            Pbrt1.append(Pbrt)
            Pdrk1.append(Pdrk)
            P_up1.append(P_up)
            P_lp1.append(P_lp)
            IPR1.append( model1.getIPR() )
            IPR2.append( model2.getIPR() )
            print(it,Pmol1[-1],Pmol2[-1],Pmol1[-1]-Pmol2[-1])
            # print("{t}\t{Pcav}".format(t=it*dt,Pcav=Pcav1[-1]) )

    fpop = open('pop.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fpop.write("{t}\t{P1}\t{P2}\t{Pbrt1}\t{Pdrk1}\t{P_up1}\t{P_lp1}\t{Prad_tot}\t{Pdamp_tot}\t{IPR1}\t{IPR2}\n".format(t=times[it],
                    P1=Pmol1[it],P2=Pmol2[it],
                    Pbrt1=Pbrt1[it],Pdrk1=Pdrk1[it],
                    P_up1=P_up1[it],P_lp1=P_lp1[it],
                    Prad_tot=Prad_tot[it],Pdamp_tot=Pdamp_tot[it],
                    IPR1=IPR1[it],IPR2=IPR2[it]))
    frad = open('rad.dat'+sys.argv[-1], 'w')
    for j in range(len(model2.Erad)):
        frad.write(str(round(model2.Erad[j],4)))
        for i in range(len(Prad)):
            frad.write('\t'+'{:2.10f}'.format(Prad[i][j][0]))
        frad.write('\n')
    fdamp = open('damp.dat'+sys.argv[-1], 'w')
    for j in range(len(model2.Erad)):
        fdamp.write(str(round(model2.Erad[j],4)))
        for i in range(len(Pdamp)):
            fdamp.write('\t'+'{:2.10f}'.format(Pdamp[i][j][0]))
        fdamp.write('\n')

    # if round(Wdrv+Wgrd,2) in [-0.5,0.0,0.5]:
    if plotResult:
        Wdrv = DriveParam['DriveFrequency']
        fig, ax= plt.subplots(1,5, figsize=(16.0,3.0))
        ax[0].semilogy(times,Pmol1,'-', lw=2, label='Pmol1, Wd='+str(round(Wdrv+Wgrd,2)))
        ax[0].semilogy(times,Pmol2,'-', lw=2, label='Pmol2, Wd='+str(round(Wdrv+Wgrd,2)))
        ax[2].semilogy(times,Pcav1,'-', lw=2, label='Pcav, Wd='+str(round(Wdrv+Wgrd,2)))
        ax[3].semilogy(times,P_up1,'-.', lw=2, label='P_up, Wd='+str(round(Wdrv+Wgrd,2)))
        ax[3].semilogy(times,P_lp1,'--', lw=2, label='P_lp, Wd='+str(round(Wdrv+Wgrd,2)))
        ax[4].semilogy(times,Pbrt1,'-', lw=2, label='Pbrt, Wd='+str(round(Wdrv+Wgrd,2)))
        ax[4].semilogy(times,Pdrk1,'-', lw=2, label='Pdrk, Wd='+str(round(Wdrv+Wgrd,2)))
        # ax[2].plot(times,Pgrd1,'-', lw=2, label='Pgrd', alpha=0.7)
        ax[0].set_xlabel("$t$")
        ax[0].set_ylabel("$P$")
        ax[0].legend()
        ax[2].legend()
        ax[3].legend()
        ax[4].legend()
        
        Prad = np.array(Prad)        
        Pdamp = np.array(Pdamp)
        Xgrid, Ygrid = np.meshgrid(model2.Erad, times)
        Zgrid = Prad[:,:,0] #+ Pdamp[:,:,0]
        ax[1].contourf(Xgrid, Ygrid, Zgrid, 100, cmap=plt.cm.jet)
        ax[1].set_xlabel(r"$\omega_\alpha$")
        ax[1].set_ylabel("$t$")
        plt.show()

