import numpy as np
from matplotlib import pyplot as plt

from trajectory import Trajectory_SSHmodel

param_script="""
This test case is designed to reproduce: PRL 96, 086601 (2006)
We convert the unit as follows:
Length: angstrom (A)    = 10^-10 m
Time:   picosecond (ps) = 10^-12 s
Energy: Joul/mol = kg/mol*m^2/s^2 

Mass:   Joul/mol*ps^2/A^2
        1 amu = 1.66*10^-27 (kg) * 6.022*10^23 (mol) = 9.9965*10^-4 ~ 10^-3 kg/mol
              = 10^-3 Joul/mol*s^2/m^2 = 10^-3 * (10^12)^2 / (10^10)^2 Joul/mol*ps^2/A^2 
              = 10.0 Joul/mol*ps^2/A^2

K:      1 amu/ps^2  = 1.66*10^-27 (kg) * 6.022*10^23 (mol) / (10^-12)^2 (ps) = 9.9965*10^20 ~ 10^21 kg/mol/s^2
                    = 10^21 Joul/mol/m^2 = 10^21 / (10^10)^2 Joul/mol/A^2
                    = 10.0 Joul/mol/A^2

hbar:   1.0546*10^-34 Joul*s = 1.0546*10^-34 * 6.022*10^23 (mol) * 10^12 (ps) 
                             ~ 63.508 Joul/mol*ps

kB:     1.38*10^-23 Joul/K = 1.38*10^-23 * 6.022*10^23 (mol)
                           ~ 8.3 Joul/mol/K 

cm^-1   = hc    = 6.626*10^-34 (Joul*s) * 2.997*10^8 (m/s)  = 1.9864*10^-25 Joul*m
                = 1.9864*10^-25 * 6.022*10^23 (mol) * 10^2 (cm) 
                = 11.962 Joul/mol*cm


Input parameter:
    mass =  {mass} Joul/mol(ps/A)^2
    K =     {Kconst}  Joul/mol/A^2
    t =     {staticCoup} 1/ps
    alpha = {dynamicCoup} 1/ps/A
    kBT =   {kBT} Joul/mol
"""

#fundamental constants 
hbar   = 63.508
kB     = 8.3

mass = 1000.0 #amu
Kconst = 14500.0 #amu/ps^2
staticCoup = 300.0 #t: transfer integral (cm^{-1})
dynamicCoup = 995.0 #alpha (cm^{-1}/A)
temperature  = 150.0 #temperature (K)

mass = mass * 10.0  # amu to Joul/mol(ps/A)^2
Kconst = Kconst * 10.0 #amu/ps^2 to Joul/mol/A^2
staticCoup = staticCoup * 11.962/hbar #cm^-1 to 1/ps
dynamicCoup = dynamicCoup * 11.962/hbar #cm^-1/A to 1/ps/A
kBT = temperature * kB  #K to J/mol

dt = 0.001 #ps
Ntimes = 5000
Nskip = 150
Nmol = 601 

print(param_script.format(mass=mass,Kconst=Kconst,staticCoup=staticCoup,dynamicCoup=dynamicCoup,kBT=kBT))

traj = Trajectory_SSHmodel(Nmol)
traj.initialHamiltonian(staticCoup,dynamicCoup)
traj.initialGaussian(kBT,mass,Kconst)
traj.updateHmol()
traj.initialState(hbar,kBT,most_prob=True)
# traj.Cj[(Nmol+1)/2] = np.sqrt(0.5)
# traj.Cj[(Nmol+1)/2+1] = np.sqrt(0.5)
print("Probability=\t{prob}".format(prob=traj.Prob))

times = []
Xj_list = []
Vj_list = []
Ej_list = []
Displacement_list = []

slice_list = []
distr_list = []

E0 = traj.getEnergy()
for it in range(Ntimes):
    # if it%1000==0:
    traj.velocityVerlet(dt)
    traj.updateHmol()
    traj.propagateCj(dt)



    if it%Nskip==0:
        time = it*dt
        Displacement = traj.getDisplacement()
        Population = np.linalg.norm(traj.Cj)
        times.append(time)
        print("{t}\t{d}\t{dp}".format(t=time,d=Displacement, dp=Population-1.0))
        Displacement_list.append(Displacement)
        Ej_list.append(traj.getEnergy())
        Xj_list.append(traj.Xj)
        Vj_list.append(traj.Vj)
        distr_list.append(np.abs(traj.Cj)**2)

Xj_list, Vj_list = np.array(Xj_list).T, np.array(Vj_list).T
distr_list = np.array(distr_list)
Displacement_list = np.array(Displacement_list)

fig, ax = plt.subplots(4)
ax[0].plot(times,Ej_list/E0-1.0,label='classical energy conservation')
# ax[0].plot(times,Pop_list-1.0,label='quantum population conservation')
ax[0].legend()

for j in range(Nmol):
    ax[1].plot(Xj_list[j],Vj_list[j])

for it in range(len(times)):
    ax[2].plot(distr_list[it])

ax[3].plot(times,Displacement_list)
plt.show()

