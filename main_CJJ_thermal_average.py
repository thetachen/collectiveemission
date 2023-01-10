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

# model1.initialCj_Polariton(-1) # initialize before disorder

# model1.initialXjVj_Gaussian(kBT,mass,Kconst)
# model1.updateNeighborHarmonicOscillator(staticCoup,dynamicCoup)
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

initial_state=0
# EigEng = model1.initialCj_Eigenstate_Forward(Wmol,Vndd,initial_state=initial_state)
if Vcav==0.0:
    EigEng, EigSta = model1.initialCj_Eigenstate_Hmol(initial_state=initial_state)
else:
    EigEng, EigSta = model1.initialCj_Eigenstate_Hcavmol(initial_state=initial_state)
# cos = np.real(EigEng)/2.0/Vndd
# sin2 = (4.0*Vndd**2-np.real(EigEng)**2)/4.0/Vndd**2
# print(EigEng)
# ###Find the states that hybrid the cavity photon
# ind_cav = np.where(np.abs(EigSta[0,:])>0.01)
# print(ind_cav)
# sin2[ind_cav]=0.0
# print(sin2)

### Stationary evaluation of CJJ (for static disorder)
# fig, ax= plt.subplots(3,figsize=(8.0,12.0))
# kBT_list = 10.0**np.linspace(-3,1,11)
# CJJ_avg_list = []
# for kBT in kBT_list:
if useDynamicNeighborDisorder or useDynamicDiagonalDisorder:
    print("Cannot use the stationary evaluation of CJJ")
else:
    print("Use the stationary evaluation of CJJ")
    if useNodisorder:
        Ntimes = 2 
    times = np.arange(Ntimes)*dt
    if Vcav==0.0:
        Nsize = Nmol
        Jmol = model1.Jt0[model1.Imol:model1.Imol+model1.Nmol,model1.Imol:model1.Imol+model1.Nmol]
    else:
        Nsize = Nmol +1 
        Jmol = model1.Jt0[model1.Icav:model1.Imol+model1.Nmol,model1.Icav:model1.Imol+model1.Nmol]
    J0mn = np.dot(np.conj(EigSta).T,np.dot(Jmol,EigSta))
    
    CJJ_avg = np.zeros(Ntimes,complex)
    Partition = 0.0
    for m in range(Nsize):
        CJJ_ana = np.zeros(Ntimes,complex)
        for n in range(Nsize):
            CJJ_ana = CJJ_ana + np.exp(1j*(EigEng[m]-EigEng[n])*times)*np.abs(J0mn[m,n])**2
        CJJ_avg = CJJ_avg + CJJ_ana * np.exp(-EigEng[m]/kBT)
        Partition = Partition + np.exp(-EigEng[m]/kBT)
    CJJ_avg = CJJ_avg/Partition

    if not plotResult:
        fcorr = open('CJJ_therm.dat'+sys.argv[-1], 'w')
        for it in range(len(times)):
            fcorr.write("{t}\t{Corr_real}\t{Corr_imag}\n".format(t=times[it],Corr_real=np.real(CJJ_avg)[it],Corr_imag=np.imag(CJJ_avg)[it]))

        feigen = open('Eigen.dat'+sys.argv[-1], 'w')
        for i in range(len(EigEng)):
            feigen.write("{EigEng}\n".format(EigEng=EigEng[i]))

    exit()

# sites = np.arange(Nmol)
# ax[0].plot(sites,np.real(model1.Cj)[2:,0], '-o',lw=2, alpha=0.7)
# ax[0].plot(sites,np.imag(model1.Cj)[2:,0], '-o',lw=2, alpha=0.7)
# J0Cj = np.dot(model1.Jt0,model1.Cj)/Vndd/-1j/np.sin((initial_state)*2*np.pi/Nmol)
# ax[0].plot(sites,np.real(J0Cj)[2:,0], '-x',lw=2, alpha=0.7)
# ax[0].plot(sites,np.imag(J0Cj)[2:,0], '-x',lw=2, alpha=0.7)

# ax[1].plot(sites,np.real(EigEng), '-x',lw=2)
# ax[1].plot(sites,np.imag(EigEng), '-+',lw=2)
# ax[1].plot(sites,np.sort(Vndd*np.cos(sites*2*np.pi/Nmol)), '-x',lw=2)
# ax[1].plot(sites,Vndd*np.sin(sites*2*np.pi/Nmol), '-+',lw=2)
# plt.show()

times = []
Pmol1, Pmol2 = [], []
IPR1, IPR2 = [], []

distr_list = []
Displacement_list = []
Correlation_list = []
Current_list = []

for it in range(Ntimes):
    if useDynamicNeighborDisorder:
        model1.updateNeighborDynamicDisorder(DeltaNN,TauNN,dt)
    if useDynamicDiagonalDisorder:
        model1.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)

    Javg, CJJ = model1.getCurrentCorrelation()

    model1.propagateCj_RK4(dt)
    model1.propagateJ0Cj_RK4(dt)
    
    if it%Nskip==0:
        times.append( it*dt )
        Pmol1.append( model1.getPopulation_system()  )
        # Pmol2.append( model2.getPopulation_system() )

        IPR1.append( model1.getIPR() )
        # IPR2.append( model2.getIPR() )

        distr = np.abs(model1.Cj[model1.Imol:model1.Imol+model1.Nmol])**2
        distr_list.append(distr[:,0])
        Displacement_list.append(model1.getDisplacement())
        Correlation_list.append(CJJ)
        Current_list.append(Javg)
        if printOutput:
            print("{t}\t{d}\t{dP}".format(t=it*dt,d=model1.getDisplacement(),dP=model1.getPopulation_system()))
                                                # dE=model1.getEnergy()-E0 ))

if not plotResult:
    # write to output 
    fpop = open('Pmol.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fpop.write("{t}\t{Pmol}\n".format(t=times[it],Pmol=Pmol1[it]))

    fdis = open('Displacement.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fdis.write("{t}\t{Displacement}\n".format(t=times[it],Displacement=Displacement_list[it]))

    fcorr = open('Correlation.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fcorr.write("{t}\t{Corr_real}\t{Corr_imag}\n".format(t=times[it],Corr_real=np.real(Correlation_list[it]),Corr_imag=np.imag(Correlation_list[it])))

    # fwfn = open('wfn.dat'+sys.argv[-1], 'w')
    # for j in range(model1.Nmol):
    #     fwfn.write(str(j))
    #     for i in range(len(distr_list)):
    #         fwfn.write('\t'+'{:2.10f}'.format(distr_list[i][j]))
    #     fwfn.write('\n')

if plotResult:
    from scipy import stats, special

    distr_list = np.array(distr_list)

    fig, ax= plt.subplots(1,7, figsize=(18.0,3.0))
    ax[0].plot(times,Pmol1, '-r', lw=2, label='Q matrix', alpha=0.7)
    # ax[0].plot(times,Pmol2, '-k', lw=2, label='Explicit', alpha=0.7)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("$P_{mol}$")
    ax[0].legend()

    # # cuts = [1.0,9.0]
    # cuts = [0.0,Ntimes*dt]
    # cut1=np.argmin(np.abs(np.array(times)-cuts[0]))
    # cut2=np.argmin(np.abs(np.array(times)-cuts[1]))

    # res = stats.linregress(np.log10(times[cut1:cut2]), np.log10(np.array(Displacement_list)[cut1:cut2]))
    # slope = res.slope
    # intercept = res.intercept
    # print(slope)

    ax[1].plot(times,np.array(Displacement_list),label='numerical')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('Displacement')

    Xgrid = np.array(range(len(distr_list[0])))
    Ygrid = times
    Xgrid, Ygrid = np.meshgrid(Xgrid, Ygrid)
    Zgrid = distr_list
    CS = ax[2].contourf(Xgrid, Ygrid, Zgrid, 100, cmap=plt.cm.jet)
    ax[2].set_xlabel('sites')
    ax[2].set_ylabel('time')

    times = np.array(times) #+dt*Nskip
    
    Xmean = np.zeros(len(times))
    Xvar = np.zeros(len(times))

    pop_ana_list = []
    for x_ind in range(Nmol):
        index = x_ind-int(Nmol/2)
        ln = ax[3].plot(times,distr_list[:,x_ind],alpha=0.5,label='numerical')
        lncolor = ln[0].get_color()
        
        if Vcav==0.0:
            pop_ana = special.jv(index,2*np.abs(Vndd)*times)**2
        else:
            Omega_ana = 0.5*(Wcav + Wmol - 2.0*np.abs(Vndd))
            Delta_ana = 0.5*(Wcav - Wmol + 2.0*np.abs(Vndd))
            energy_gap = np.sqrt(Delta_ana**2 + np.abs(Vcav)**2*Nmol)
            coef_ana = np.exp(-1j*Wmol*times)*(1j**index)*special.jv(index,2*np.abs(Vndd)*times) \
                    - np.exp(-1j*(Wmol-2*np.abs(Vndd))*times)/Nmol \
                    + np.exp(-1j*Omega_ana*times)*(np.cos(energy_gap*times)+1j*Delta_ana/energy_gap*np.sin(energy_gap*times))/Nmol
            pop_ana = np.abs(coef_ana)**2

        pop_ana_list.append(pop_ana)
        ax[3].plot(times,pop_ana,'--',color=lncolor,lw=2,alpha=0.5,label='analytical')
    ax[3].set_xlabel('time')
    ax[3].set_ylabel('population')

    ### calculate the displacement using analytical result. 
    pop_ana_list = np.array(pop_ana_list).T
    displace_ana = []
    for it in range(len(times)):
        Rj = np.array(range(Nmol))
        R =  np.abs( np.sum( Rj       *pop_ana_list[it]) ) 
        R2 = np.abs( np.sum((Rj-R)**2 *pop_ana_list[it]) )     
        displace_ana.append(R2)
    ax[1].plot(times,displace_ana,'--k',label="direct Bessel summation")
    ax[1].plot(times,0.5*(2*np.abs(Vndd)*np.array(times))**2,':',label='analytical (no cavity)')
    ax[1].legend()

    W,U = np.linalg.eigh(model1.Ht0)
    eigen_num = np.delete(W, np.where(W == Wgrd))
    if Vcav==0.0:
        eigen_num = np.delete(eigen_num, np.where(eigen_num == Wcav))
        # print(eigen_num)
        eigen_ana = Wmol-2.0*np.abs(Vndd)*np.cos(2*np.pi*np.arange(1,Nmol+1)/Nmol)
        print(np.isclose(eigen_num,np.sort(eigen_ana)))
    else:
        eigen_ana = Wmol-2.0*np.abs(Vndd)*np.cos(2*np.pi*np.arange(1,Nmol+1)/Nmol)
        eigen_ana = np.delete(eigen_ana, np.where(eigen_ana == Wmol-2.0*np.abs(Vndd)))
        energy_gap = np.sqrt(0.25*(Wcav - Wmol + 2.0*np.abs(Vndd))**2+np.abs(Vcav)**2*Nmol)
        eigen_ana = np.append(eigen_ana,0.5*(Wcav + Wmol - 2.0*np.abs(Vndd)) + energy_gap)
        eigen_ana = np.append(eigen_ana,0.5*(Wcav + Wmol - 2.0*np.abs(Vndd)) - energy_gap)
        print(np.isclose(eigen_num,np.sort(eigen_ana)))
    print(np.sort(eigen_ana)[initial_state],eigen_num[initial_state])

    if Vcav!=0.0:# WITH CAVITY
        Wcav_max = 10.0
        Wcav_list = np.linspace(-Wcav_max, Wcav_max, num=101)
        eigen_ana_list = []
        for iWcav in range(len(Wcav_list)):
            eigen_ana = Wmol-2.0*np.abs(Vndd)*np.cos(2*np.pi*np.arange(1,Nmol+1)/Nmol)
            eigen_ana = np.delete(eigen_ana, np.where(eigen_ana == Wmol-2.0*np.abs(Vndd)))
            energy_gap = np.sqrt(0.25*(Wcav_list[iWcav] - Wmol + 2.0*np.abs(Vndd))**2+np.abs(Vcav)**2*Nmol)
            eigen_ana = np.append(eigen_ana,0.5*(Wcav_list[iWcav] + Wmol - 2.0*np.abs(Vndd)) + energy_gap)
            eigen_ana = np.append(eigen_ana,0.5*(Wcav_list[iWcav] + Wmol - 2.0*np.abs(Vndd)) - energy_gap)
            eigen_ana_list.append(eigen_ana)
        eigen_ana_list = np.array(eigen_ana_list).T
        for i in range(len(eigen_ana)):
            ax[4].plot(Wcav_list,eigen_ana_list[i])
        ax[4].plot(Wcav_list,(Wmol-2.0*np.abs(Vndd))*np.ones(len(Wcav_list)),'--k')
        ax[4].plot(Wcav_list,Wcav_list,'--')
        ax[4].set_xlabel('$\omega_c$')
        ax[4].set_ylabel('eigenenergy')
        ax[4].axvline(x=Wcav,ls='--')

        # ax[5].plot(times,np.array(displace_ana)-0.5*(2*np.abs(Vndd)*np.array(times))**2)
        # Delta_ana = 0.5*(Wcav - Wmol + 2.0*np.abs(Vndd))
        # shift = (Nmol-1)*((Nmol-1)+1)*(2*(Nmol-1)+1)/6/(Nmol-1)**2 \
        #         *(Delta_ana**2/(Delta_ana**2+np.abs(Vcav)**2*Nmol)+1.0)
        # ax[5].plot(times,np.ones(len(times))*shift,'--')
        # ax[5].set_xlabel('time')
        # ax[5].set_ylabel('$\Delta$ displacement')
        # ax[1].plot(times,0.5*(2*np.abs(Vndd)*np.array(times))**2+shift,':')
        ax[5].plot(times,np.real(Correlation_list),'-b')
        ax[5].plot(times,np.imag(Correlation_list),'-r')
        # ax[5].axhline(y=(4.0*Vndd**2-np.real(EigEng[initial_state])**2),ls='--')
        ax[5].axhline(y=(4.0*Vndd**2)*sin2[initial_state],ls='--')
        ax[5].set_xlabel('time')
        ax[5].set_ylabel('current correlation')
        ax[6].plot(times,np.real(Current_list),'-b')
        ax[6].plot(times,np.imag(Current_list),'-r')
        ax[6].axhline(y=2.0*np.sqrt(Vndd**2-np.real(EigEng[initial_state])**2),ls='--')
        ax[6].set_xlabel('time')
        ax[6].set_ylabel('current')
    else:# WITHOUT CAVITY
        ax[5].plot(times,np.real(Correlation_list),'-b')
        ax[5].plot(times,np.imag(Correlation_list),'-r')
        ax[5].axhline(y=4.0*(Vndd**2-np.real(EigEng[initial_state])**2),ls='--')
        ax[5].set_xlabel('time')
        ax[5].set_ylabel('current correlation')
        ax[6].plot(times,np.real(Current_list),'-b')
        ax[6].plot(times,np.imag(Current_list),'-r')
        ax[6].axhline(y=2.0*np.sqrt(Vndd**2-np.real(EigEng[initial_state])**2),ls='--')
        ax[6].set_xlabel('time')
        ax[6].set_ylabel('current')

    
    # ax[1].set_xscale('log')
    # ax[1].set_yscale('log')
    plt.tight_layout()
    # plt.show()  

times = np.array(times)
Jmol = model1.Jt0[model1.Imol:model1.Imol+model1.Nmol,model1.Imol:model1.Imol+model1.Nmol]
J0mn = np.dot(np.conj(EigSta).T,np.dot(Jmol,EigSta))
CJJ_ana = np.zeros(len(times),complex)
for n in range(Nmol):
    CJJ_ana = CJJ_ana + np.exp(1j*(EigEng[initial_state]-EigEng[n])*times)*np.abs(J0mn[initial_state,n])**2
ax[5].plot(times,np.real(CJJ_ana),'-xb')
ax[5].plot(times,np.imag(CJJ_ana),'-xr')
plt.show()  