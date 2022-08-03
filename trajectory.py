import numpy as np
from matplotlib import pyplot as plt

class Trajectory():

    def __init__(self,Nmol,mass,Kconst,staticCoup,dynamicCoup):
        self.Nmol = Nmol
        self.mass = mass
        self.Kconst = Kconst
        self.staticCoup = staticCoup
        self.dynamicCoup = dynamicCoup
        self.Hmol = np.zeros((Nmol,Nmol),complex)
        self.Hmol_dt = np.zeros((Nmol,Nmol),complex)
        self.Cj = np.zeros((Nmol,1),complex)
        self.Xj = np.zeros(Nmol)
        self.Vj = np.zeros(Nmol)
        self.Rj = np.array(range(Nmol))
        np.random.seed()

    def initialGaussian(self,kBT):
        self.Xj = np.random.normal(0.0, kBT/self.Kconst, self.Nmol)
        self.Vj = np.random.normal(0.0, kBT/self.mass,   self.Nmol)

    def initialState(self,kBT):
        W,U = np.linalg.eigh(self.Hmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]

        self.Prob = np.exp(-W*hbar/kBT)
        self.Prob = self.Prob/np.sum(self.Prob)

        rand = np.random.random()
        
        Prob_cum = np.cumsum(self.Prob)
        initial_state = 0
        while rand > Prob_cum[initial_state]:
            initial_state += 1
        initial_state -= 1

        # print(rand, Prob_cum[initial_state],Prob_cum[initial_state+1])

        self.Cj = U[:,initial_state]

        var_list = []
        for i in range(Nmol):
            R =  np.abs( np.sum(self.Rj    *np.abs(U[:,i].T)**2) ) 
            R2 = np.abs( np.sum((self.Rj-R)**2 *np.abs(U[:,i].T)**2) ) 
            var_list.append(R2)
        print(min(var_list))
        
    def velocityVerlet(self,dt):
        """
        We use the algorithm with eliminating the half-step velocity
        https://en.wikipedia.org/wiki/Verlet_integration
        """
        # 1: calculate Aj(t)
        Aj = -self.Kconst/self.mass * self.Xj
        for j in range(1,self.Nmol-1):
            Aj[j] = Aj[j] -self.dynamicCoup/self.mass* \
                    ( 2*np.real(np.conj(self.Cj[j])*self.Cj[j-1]) \
                    - 2*np.real(np.conj(self.Cj[j])*self.Cj[j+1]))
        Aj[0] = Aj[0] -self.dynamicCoup/self.mass* \
                ( 2*np.real(np.conj(self.Cj[0])*self.Cj[-1]) \
                - 2*np.real(np.conj(self.Cj[0])*self.Cj[1]))
        Aj[-1] = Aj[-1] -self.dynamicCoup/self.mass* \
                    ( 2*np.real(np.conj(self.Cj[-1])*self.Cj[-2]) \
                    - 2*np.real(np.conj(self.Cj[-1])*self.Cj[0]))
        # 2: calculate Xj(t+dt)
        self.Xj = self.Xj + self.Vj*dt + 0.5*dt**2*Aj
        # 3: calculate Aj(t+dt)+Aj(t)
        Aj = Aj -self.Kconst/self.mass * self.Xj
        # 4: calculate Vj(t+dt)
        self.Vj = self.Vj + 0.5*dt*Aj

    def updateHmol(self):
        for j in range(self.Nmol-1):
            self.Hmol[j,j+1] = -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.Hmol[j+1,j] = -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.Hmol_dt[j,j+1] = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])
            self.Hmol_dt[j+1,j] = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])

        self.Hmol[0,-1] = -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.Hmol[-1,0] = -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.Hmol_dt[0,-1] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])
        self.Hmol_dt[-1,0] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])


    def propagateCj(self,dt):
        self.Cj = self.Cj - 1j*dt*np.dot(self.Hmol,self.Cj) \
                  -0.5*dt**2*np.dot(self.Hmol,np.dot(self.Hmol,self.Cj)) \
                  -0.5*1j*dt**2*np.dot(self.Hmol_dt,self.Cj)


    def getEnergy(self):
        return 0.5*self.mass*np.linalg.norm(self.Vj)**2 + 0.5*self.Kconst*np.linalg.norm(self.Xj)**2

    def getDisplacement(self):
        # print(np.sum(np.abs(self.Cj.T)**2))
        # R2 = np.abs( np.sum(self.Rj**2 *np.abs(self.Cj.T)**2) ) 
        R =  np.abs( np.sum(self.Rj    *np.abs(self.Cj.T)**2) ) 
        R2 = np.abs( np.sum((self.Rj-R)**2 *np.abs(self.Cj.T)**2) ) 

        print(R2,R)
        return (R2)


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

mass = 250.0 #amu
Kconst = 14500.0 #amu/ps^2
staticCoup = 300.0 #t: transfer integral (cm^{-1})
dynamicCoup = 995.0 #alpha (cm^{-1}/A)
temperature  = 150.0 #temperature (K)

mass = mass * 10.0  # amu to Joul/mol(ps/A)^2
Kconst = Kconst * 10.0 #amu/ps^2 to Joul/mol/A^2
staticCoup = staticCoup * 11.962/hbar #cm^-1 to 1/ps
dynamicCoup = dynamicCoup * 11.962/hbar #cm^-1/A to 1/ps/A
kBT = temperature * kB #K to J/mol

dt = 0.001 #ps
Ntimes = 5000
Nskip = 150
Nmol = 601 

print(param_script.format(mass=mass,Kconst=Kconst,staticCoup=staticCoup,dynamicCoup=dynamicCoup,kBT=kBT))

fig, ax = plt.subplots(3)

traj = Trajectory(Nmol,mass,Kconst,staticCoup,dynamicCoup)

traj.initialGaussian(kBT)
traj.updateHmol()
traj.initialState(kBT)
# traj.Cj[(Nmol+1)/2] = np.sqrt(0.5)
# traj.Cj[(Nmol+1)/2+1] = np.sqrt(0.5)

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
        print("{t}\t{d}\t{dp}\n".format(t=time,d=Displacement, dp=Population-1.0))
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

