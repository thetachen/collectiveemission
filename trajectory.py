import numpy as np
from copy import deepcopy
class Trajectory_SSHmodel():

    def __init__(self,Nmol,seed=None):
        self.Nmol = Nmol
        self.Hmol = np.zeros((Nmol,Nmol),complex)
        self.Hmol_dt = np.zeros((Nmol,Nmol),complex)
        self.Cj = np.zeros((Nmol,1),complex)
        self.Xj = np.zeros(Nmol)
        self.Vj = np.zeros(Nmol)
        self.Rj = np.array(range(Nmol))
        np.random.seed(seed)

    def initialHamiltonian(self,staticCoup,dynamicCoup):
        self.staticCoup = staticCoup
        self.dynamicCoup = dynamicCoup

        for j in range(self.Nmol-1):
            self.Hmol[j,j+1] = -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.Hmol[j+1,j] = -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.Hmol_dt[j,j+1] = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])
            self.Hmol_dt[j+1,j] = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])

        self.Hmol[0,-1] = -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.Hmol[-1,0] = -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.Hmol_dt[0,-1] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])
        self.Hmol_dt[-1,0] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])

    def initialGaussian(self,kBT,mass,Kconst):
        self.mass = mass
        self.Kconst = Kconst

        self.Xj = np.random.normal(0.0, kBT/self.Kconst, self.Nmol)
        self.Vj = np.random.normal(0.0, kBT/self.mass,   self.Nmol)

    def initialState(self,hbar,kBT,most_prob=False):
        """
        Choose the initial state from the set of eigenfunctions based on Boltzman distribution exp(-E_n/kBT)
        """
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
        if most_prob:   
            initial_state = np.argmax(self.Prob) # most probable state
        
        self.Cj = U[:,initial_state]
        self.Prob = self.Prob[initial_state]

        # var_list = []
        # for i in range(self.Nmol):
        #     R =  np.abs( np.sum(self.Rj    *np.abs(U[:,i].T)**2) ) 
        #     R2 = np.abs( np.sum((self.Rj-R)**2 *np.abs(U[:,i].T)**2) ) 
        #     var_list.append(R2)
        # print(min(var_list))
        
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
        return R2

class SingleExcitationWithCollectiveCoupling():

    def __init__(self,Nmol,Nrad,seed=None):
        self.Nmol = Nmol
        self.Nrad = Nrad
        np.random.seed(seed)

    def initialHamiltonian_Radiation(self,Wgrd,Wmol,Vrad,Wmax,damp,useQmatrix=False):
        """
        Construct the Hamiltonian in the form of 
        Ht0 = 
            | grd     | mol     | rad 
        grd | Hgrd    |         |         
        mol | Vmolgrd | Hmol    |
        rad | Vradgrd | Vradmol | Hrad
        """
        self.Wmol = Wmol
        self.useQmatrix = useQmatrix
        self.damp = damp
        self.Erad = np.zeros(self.Nrad)

        Hgrd = np.eye(1) * Wgrd
        Hmol = np.eye(self.Nmol) * Wmol
        Hrad = np.zeros((self.Nrad,self.Nrad),complex)

        Vradmol = np.zeros((self.Nrad,self.Nmol),complex)
        Vmolgrd = np.ones((self.Nmol,1),complex)
        Vradgrd = np.zeros((self.Nrad,1),complex)


        # Construct the molecule-radiation coupling
        Gamma = 0.0
        for j in range(self.Nrad):
            self.Erad[j] = ( j - (self.Nrad-1)/2 ) * Wmax *2.0/(self.Nrad-1)
            Hrad[j,j] = self.Erad[j] - 1j*self.damp
            for i in range(self.Nmol):
                Vradmol[j,i] = Vrad # * (Wrad_width**2) / ( Erad[j]**2 + Wrad_width**2 )
            Gamma += -2.0*1j*(Vradmol[j,0]**2)/Hrad[j,j] # set to be the same: 0
        #Gamma = 1j*Gamma*(Vrad**2)
        self.Gamma = np.real(Gamma)
        # print(self.Gamma)
        
        drive = 0.0
        if useQmatrix:
            Qmol = np.ones((self.Nmol,self.Nmol))
            self.Ht0 = np.vstack(( np.hstack(( Hgrd,          Vmolgrd.T*drive               )),
                                   np.hstack(( Vmolgrd*drive, Hmol - 1j*(self.Gamma/2)*Qmol )) ))        
        else:
            self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vmolgrd.T*drive,    Vradgrd.T    )),
                                    np.hstack(( Vmolgrd*drive, Hmol,               Vradmol.T    )),
                                    np.hstack(( Vradgrd,       Vradmol,            Hrad         )) ))
        self.Ht = deepcopy(self.Ht0)

        self.Imol = 1
        self.Irad = self.Nmol+1

    def initialHamiltonian_Cavity(self,Wgrd,Wcav,Wmol,Vcav,Vrad,Wmax,damp,useQmatrix=False):
        """
        Construct the Hamiltonian in the form of 
        Ht0 = 
            | grd     | cav     | mol     | rad 
        grd | Hgrd    |         |         |
        cav | Vcavgrd | Hcav    |         |  
        mol | Vmolgrd | Vmolcav | Hmol    |
        rad | Vradgrd | Vradcav | Vradmol | Hrad
        """
        self.Wmol = Wmol
        self.useQmatrix = useQmatrix
        self.damp = damp
        self.Erad = np.zeros(self.Nrad)

        Hgrd = np.eye(1) * Wgrd
        Hcav = np.eye(1) * Wcav
        Hmol = np.eye(self.Nmol) * Wmol
        Hrad = np.zeros((self.Nrad,self.Nrad),complex)

        Vradmol = np.zeros((self.Nrad,self.Nmol),complex)
        Vmolgrd = np.ones((self.Nmol,1),complex)
        Vradgrd = np.zeros((self.Nrad,1),complex)
        Vcavgrd = np.zeros((1,1),complex)
        Vmolcav = np.ones((self.Nmol,1),complex) * Vcav
        Vradcav = np.zeros((self.Nrad,1),complex)

        # Construct the molecule-radiation coupling
        Gamma = 0.0
        for j in range(self.Nrad):
            self.Erad[j] = ( j - (self.Nrad-1)/2 ) * Wmax *2.0/(self.Nrad-1)
            Hrad[j,j] = self.Erad[j] - 1j*self.damp
            for i in range(self.Nmol):
                Vradmol[j,i] = Vrad # * (Wrad_width**2) / ( Erad[j]**2 + Wrad_width**2 )
            Gamma += -2.0*1j*(Vradmol[j,0]**2)/Hrad[j,j] # set to be the same: 0
        #Gamma = 1j*Gamma*(Vrad**2)
        self.Gamma = np.real(Gamma)
        # print(self.Gamma)
        
        drive = 0.0
        if useQmatrix:
            Qmol = np.ones((self.Nmol,self.Nmol))
            self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vcavgrd.T,     Vmolgrd.T*drive               )),
                                    np.hstack(( Vcavgrd,       Hcav,          Vmolcav.T                     )),
                                    np.hstack(( Vmolgrd*drive, Vmolcav,       Hmol - 1j*(self.Gamma/2)*Qmol )) ))
        else:
            self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vcavgrd.T,     Vmolgrd.T*drive,    Vradgrd.T )),
                                    np.hstack(( Vcavgrd,       Hcav,          Vmolcav.T,          Vradcav.T )),
                                    np.hstack(( Vmolgrd*drive, Vmolcav,       Hmol,               Vradmol.T )),
                                    np.hstack(( Vradgrd,       Vradcav,       Vradmol,            Hrad      )) ))
        self.Ht = deepcopy(self.Ht0)
        
        self.Icav = 1
        self.Imol = 2
        self.Irad = self.Nmol+2

    def updateDiagonalStaticDisorder(self,Delta):
        self.Ht = deepcopy(self.Ht0)

        self.Wstc = np.random.normal(0.0,Delta,self.Nmol) + self.Wmol
        for j in range(self.Nmol): 
            self.Ht[self.Imol+j,self.Imol+j] += self.Wstc[j]

    def updateDiagonalDynamicDisorder(self,Delta,TauC,dt):
        # simulate Gaussian process
        # cf. George B. Rybicki's note
        # https://www.lanl.gov/DLDSTP/fast/OU_process.pdf
        self.Ht = deepcopy(self.Ht0)

        if not hasattr(self, 'Wdyn'):
            self.Wdyn = np.random.normal(0.0,Delta,self.Nmol) + self.Wmol
        else:
            ri = np.exp(-dt/TauC) * (TauC>0.0)
            mean_it = ri*self.Wdyn
            sigma_it = Delta*np.sqrt(1.0-ri**2)
            self.Wdyn = np.random.normal(mean_it,sigma_it,self.Nmol) + self.Wmol
        
        for j in range(self.Nmol): 
            self.Ht[self.Imol+j,self.Imol+j] += self.Wdyn[j]

    def updateNeighborStaticDisorder(self,Delta):
        self.Ht = deepcopy(self.Ht0)

        if not hasattr(self, 'Vstc'):
            self.Vstc = np.random.normal(0.0,Delta,self.Nmol)
        
        for j in range(self.Nmol-1): 
            self.Ht[self.Imol+j,   self.Imol+j+1] += self.Vstc[j]
            self.Ht[self.Imol+j+1, self.Imol+j]   += self.Vstc[j]
        
        self.Ht[self.Imol,self.Imol+self.Nmol-1] += self.Vstc[-1]
        self.Ht[self.Imol+self.Nmol-1,self.Imol] += self.Vstc[-1]

    def updateNeighborDynamicDisorder(self,Delta,TauC,dt):
        # simulate Gaussian process
        # cf. George B. Rybicki's note
        # https://www.lanl.gov/DLDSTP/fast/OU_process.pdf
        self.Ht = deepcopy(self.Ht0)

        # if not hasattr(self, 'Vdyn'):
        #     self.Vdyn = np.random.normal(0.0,Delta,self.Nmol)
        # else:
        #     ri = np.exp(-dt/TauC) * (TauC>0.0)
        #     mean_it = ri*self.Vdyn
        #     sigma_it = Delta*np.sqrt(1.0-ri**2)
        #     self.Vdyn = np.random.normal(mean_it,sigma_it,self.Nmol)
        
        # for j in range(self.Nmol-1): 
        #     self.Ht[self.Imol+j,   self.Imol+j+1] += self.Vdyn[j]
        #     self.Ht[self.Imol+j+1, self.Imol+j]   += self.Vdyn[j]
        
        # self.Ht[self.Imol,self.Imol+self.Nmol-1] += self.Vdyn[-1]
        # self.Ht[self.Imol+self.Nmol-1,self.Imol] += self.Vdyn[-1]

        if not hasattr(self, 'Xdyn'):
            self.Xdyn = np.random.normal(0.0,1.0,self.Nmol)
        else:
            ri = np.exp(-dt/TauC) * (TauC>0.0)
            mean_it = ri*self.Xdyn
            sigma_it = np.sqrt(1.0-ri**2)
            self.Xdyn = np.random.normal(mean_it,sigma_it,self.Nmol)
        
        for j in range(self.Nmol-1): 
            self.Ht[self.Imol+j,   self.Imol+j+1] += Delta*(self.Xdyn[j+1]-self.Xdyn[j])
            self.Ht[self.Imol+j+1, self.Imol+j]   += Delta*(self.Xdyn[j+1]-self.Xdyn[j])
        
        self.Ht[self.Imol,self.Imol+self.Nmol-1] += Delta*(self.Xdyn[0]-self.Xdyn[-1])
        self.Ht[self.Imol+self.Nmol-1,self.Imol] += Delta*(self.Xdyn[0]-self.Xdyn[-1])

    def updateNeighborHarmonicOscillator(self,staticCoup,dynamicCoup):
        self.Ht = deepcopy(self.Ht0)

        if not hasattr(self, 'dHdt'):
            self.dHdt = np.zeros_like(self.Ht0)

        self.staticCoup = staticCoup
        self.dynamicCoup = dynamicCoup

        for j in range(self.Nmol-1):
            self.Ht[self.Imol+j,   self.Imol+j+1] += -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.Ht[self.Imol+j+1, self.Imol+j]   += -self.staticCoup + self.dynamicCoup * (self.Xj[j+1]-self.Xj[j])
            self.dHdt[self.Imol+j,   self.Imol+j+1] = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])
            self.dHdt[self.Imol+j+1, self.Imol+j]   = self.dynamicCoup * (self.Vj[j+1]-self.Vj[j])

        self.Ht[self.Imol,self.Imol+self.Nmol-1] += -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.Ht[self.Imol+self.Nmol-1,self.Imol] += -self.staticCoup + self.dynamicCoup * (self.Xj[0]-self.Xj[-1])
        self.dHdt[self.Imol,self.Imol+self.Nmol-1] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])
        self.dHdt[self.Imol+self.Nmol-1,self.Imol] = self.dynamicCoup * (self.Vj[0]-self.Vj[-1])

    def initialCj_Cavity(self):
        self.Cj = np.zeros((self.Nmol,1),complex)
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    np.ones((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:
            print("cannot initial Cj in the cavity state when there is no cavity")
            exit()
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Bright(self):
        self.Cj = np.ones((self.Nmol,1),complex)/np.sqrt(self.Nmol)
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    np.zeros((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Ground(self):
        self.Cj = np.zeros((self.Nmol,1),complex)
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.ones((1,1),complex),   #grd
                                    np.zeros((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:  
            self.Cj = np.vstack( (np.ones((1,1),complex),   #grd
                                    self.Cj) )                #mol          
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Random(self):      
        self.Cj = np.ones((self.Nmol,1),complex)/np.sqrt(self.Nmol)*np.exp(1j*2*np.pi*np.random.rand(self.Nmol,1))
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                    np.zeros((1,1),complex),  #cav 
                                    self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                    self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_middle(self):
        self.Cj = np.zeros((self.Nmol,1),complex)
        self.Cj[int(self.Nmol/2)] = 1.0

        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    np.zeros((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Boltzman(self,hbar,kBT,most_prob=False):
        """
        Choose the initial state from the set of eigenfunctions based on Boltzman distribution exp(-E_n/kBT)
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hmol = self.Ht[self.Imol:self.Imol+self.Nmol,self.Imol:self.Imol+self.Nmol]

        W,U = np.linalg.eigh(Hmol)
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
        if most_prob:   
            initial_state = np.argmax(self.Prob) # most probable state
        
        self.Prob = self.Prob[initial_state]

        # Initialize state vector
        self.Cj = U.T[initial_state]
        self.Cj = self.Cj[..., None] 
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                  np.zeros((1,1),complex),  #cav
                                  self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                                  self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def propagateCj_RK4(self,dt):
        ### RK4 propagation 
        K1 = -1j*np.dot(self.Ht,self.Cj)
        K2 = -1j*np.dot(self.Ht,self.Cj+dt*K1/2)
        K3 = -1j*np.dot(self.Ht,self.Cj+dt*K2/2)
        K4 = -1j*np.dot(self.Ht,self.Cj+dt*K3)
        self.Cj += (K1+2*K2+2*K3+K4)*dt/6

    def propagateCj_dHdt(self,dt):
        if not hasattr(self, 'dHdt'):
            self.dHdt = np.zeros_like(self.Ht0)
        self.Cj = self.Cj - 1j*dt*np.dot(self.Ht,self.Cj) \
                   -0.5*dt**2*np.dot(self.Ht,np.dot(self.Ht,self.Cj)) \
                   -0.5*1j*dt**2*np.dot(self.dHdt,self.Cj)

    def initialXjVj_Gaussian(self,kBT,mass,Kconst):
        self.kBT = kBT
        self.mass = mass
        self.Kconst = Kconst

        self.Xj = np.random.normal(0.0, self.kBT/self.Kconst, self.Nmol)
        self.Vj = np.random.normal(0.0, self.kBT/self.mass,   self.Nmol)
        
    def propagateXjVj_velocityVerlet(self,dt):
        """
        We use the algorithm with eliminating the half-step velocity
        https://en.wikipedia.org/wiki/Verlet_integration
        """
        # 1: calculate Aj(t)
        Aj = -self.Kconst/self.mass * self.Xj
        for j in range(1,self.Nmol-1):
            Aj[j] = Aj[j] -self.dynamicCoup/self.mass* \
                    ( 2*np.real(np.conj(self.Cj[self.Imol+j])*self.Cj[self.Imol+j-1]) \
                    - 2*np.real(np.conj(self.Cj[self.Imol+j])*self.Cj[self.Imol+j+1]))
        Aj[0] = Aj[0] -self.dynamicCoup/self.mass* \
                ( 2*np.real(np.conj(self.Cj[self.Imol+0])*self.Cj[self.Imol+self.Nmol-1]) \
                - 2*np.real(np.conj(self.Cj[self.Imol+0])*self.Cj[self.Imol+1]))
        Aj[-1] = Aj[-1] -self.dynamicCoup/self.mass* \
                    ( 2*np.real(np.conj(self.Cj[self.Imol+self.Nmol-1])*self.Cj[self.Imol+self.Nmol-2]) \
                    - 2*np.real(np.conj(self.Cj[self.Imol+self.Nmol-1])*self.Cj[self.Imol]))
        # 2: calculate Xj(t+dt)
        self.Xj = self.Xj + self.Vj*dt + 0.5*dt**2*Aj
        # 3: calculate Aj(t+dt)+Aj(t)
        Aj = Aj -self.Kconst/self.mass * self.Xj
        # 4: calculate Vj(t+dt)
        self.Vj = self.Vj + 0.5*dt*Aj

    def getPopulation_system(self):
        return np.linalg.norm(self.Cj[self.Imol:self.Imol+self.Nmol])**2

    def getPopulation_radiation(self):
        if not self.useQmatrix:
            return np.linalg.norm(self.Cj[self.Irad:])**2
        else:
            return 0.0
            
    def getPopulation_cavity(self):
        if hasattr(self, 'Icav'):
            return np.abs(self.Cj[self.Icav])**2
        else:
            return 0.0

    def getIPR(self):
        return np.linalg.norm(self.Cj[self.Imol:self.Imol+self.Nmol])**4 \
                / np.sum(np.abs(self.Cj[self.Imol:self.Imol+self.Nmol])**4)

    def getEnergy(self):
        return 0.5*self.mass*np.linalg.norm(self.Vj)**2 + 0.5*self.Kconst*np.linalg.norm(self.Xj)**2

    def getDisplacement(self):
        # print(np.sum(np.abs(self.Cj.T)**2))
        # R2 = np.abs( np.sum(self.Rj**2 *np.abs(self.Cj.T)**2) ) 
        Rj = np.array(range(self.Nmol))
        R =  np.abs( np.sum( Rj       *np.abs(self.Cj[self.Imol:self.Imol+self.Nmol].T)**2) ) 
        R2 = np.abs( np.sum((Rj-R)**2 *np.abs(self.Cj[self.Imol:self.Imol+self.Nmol].T)**2) ) 
        return R2
