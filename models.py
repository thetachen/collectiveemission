from math import gamma
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

        self.Xj = np.random.normal(0.0, np.sqrt(kBT/self.Kconst), self.Nmol)
        self.Vj = np.random.normal(0.0, np.sqrt(kBT/self.mass),   self.Nmol)

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
        R =  np.sum(self.Rj    *np.abs(self.Cj.T)**2)
        R2 = np.sum((self.Rj-R)**2 *np.abs(self.Cj.T)**2)
        return R2

class SingleExcitationWithCollectiveCoupling():

    def __init__(self,Nmol,Nrad,seed=None):
        self.Nmol = Nmol
        self.Nrad = Nrad
        np.random.seed(seed)

    def initialHamiltonian_Radiation(self,Wgrd,Wmol,Vndd,Vrad,Wmax,damp,useQmatrix=False):
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

        # Construct the nearest dipole-dipole coupling
        for j in range(self.Nmol-1): 
            Hmol[j,   j+1] = Vndd
            Hmol[j+1, j  ] = Vndd
        Hmol[0,-1] = Vndd
        Hmol[-1,0] = Vndd
        
        drive = 0.0
        if useQmatrix:
            Qmol = np.ones((self.Nmol,self.Nmol))
            self.Ht0 = np.vstack(( np.hstack(( Hgrd,          Vmolgrd.T*drive               )),
                                   np.hstack(( Vmolgrd*drive, Hmol - 1j*(self.Gamma/2)*Qmol )) ))        
            self.Hext =np.vstack(( np.hstack(( Hgrd*0.0,    Vmolgrd.T   )),
                                   np.hstack(( Vmolgrd*0.0, Hmol*0.0    )) ))        
        else:
            self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vmolgrd.T*drive,    Vradgrd.T    )),
                                    np.hstack(( Vmolgrd*drive, Hmol,               Vradmol.T    )),
                                    np.hstack(( Vradgrd,       Vradmol,            Hrad         )) ))
            self.Hext =np.vstack((  np.hstack(( Hgrd*0.0,       Vmolgrd.T,      Vradgrd.T*0.0   )),
                                    np.hstack(( Vmolgrd*0.0,    Hmol*0.0,       Vradmol.T*0.0   )),
                                    np.hstack(( Vradgrd*0.0,    Vradmol*0.0,    Hrad*0.0        )) ))
        self.Ht = deepcopy(self.Ht0)

        self.Imol = 1
        self.Irad = self.Nmol+1

    def initialHamiltonian_nonHermitian(self,Wgrd,Wcav,Wmol,Vndd,Vcav,Kcav,Gamma=0.0):
        """
        This is a combination of initialHamiltonian_Cavity_nonHermitian
        Construct the Hamiltonian in the form of 
        if Vcav ==0:
            Ht0 = 
                | grd     | mol   
            grd | Hgrd    | 
            mol | Vmolgrd | Hmol  
        else: 
            Ht0 = 
                | grd     | cav     | mol   
            grd | Hgrd    |         |       
            cav | Vcavgrd | Hcav    |         
            mol | Vmolgrd | Vmolcav | Hmol              
        """
        self.useQmatrix = True #Just to eliminate the rad part 
        self.Wmol = Wmol
        self.Gamma = Gamma

        Hgrd = np.eye(1) * Wgrd
        Hcav = np.eye(1) * Wcav
        Hmol = np.eye(self.Nmol) * Wmol
        Jmol = np.zeros((self.Nmol,self.Nmol),complex)

        Vmolgrd = np.ones((self.Nmol,1),complex)
        Vcavgrd = np.zeros((1,1),complex)
        Vmolcav = np.ones((self.Nmol,1),complex) * Vcav
        if not Kcav==0:
            for j in range(self.Nmol):
                # Vmolcav[j,0] = Vcav*np.sin(Kcav*np.pi*j/self.Nmol)
                Vmolcav[j,0] = Vcav*np.exp(-1j*(Kcav*np.pi*j/self.Nmol))
            
        # Construct the nearest dipole-dipole coupling
        for j in range(self.Nmol-1): 
            Hmol[j,   j+1] = Vndd
            Hmol[j+1, j  ] = Vndd
            Jmol[j,   j+1] = Vndd*1j
            Jmol[j+1, j  ] =-Vndd*1j
        Hmol[0,-1] = Vndd
        Hmol[-1,0] = Vndd
        Jmol[0,-1] =-Vndd*1j
        Jmol[-1,0] = Vndd*1j

        drive = 0.0
        Qmol = np.ones((self.Nmol,self.Nmol))
        if Vcav==0.0:
            self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vmolgrd.T*drive   )),
                                    np.hstack(( Vmolgrd*drive, Hmol -1j*(self.Gamma/2)*Qmol   )) ))
            
            self.Qmat = np.vstack(( np.hstack(( Hgrd*0.0,     Vmolgrd.T*0.0   )),
                                    np.hstack(( Vmolgrd*0.0,  -1j*(self.Gamma/2)*Qmol )) ))

            self.Jt0 = np.vstack((  np.hstack(( Hgrd*0.0,      Vmolgrd.T*0.0   )),
                                    np.hstack(( Vmolgrd*0.0,   Jmol )) ))
            self.Imol = 1
        else:
            self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vcavgrd.T,     Vmolgrd.T*drive   )),
                                    np.hstack(( Vcavgrd,       Hcav,          np.conj(Vmolcav).T)),
                                    np.hstack(( Vmolgrd*drive, Vmolcav,       Hmol -1j*(self.Gamma/2)*Qmol   )) ))
            
            self.Qmat = np.vstack((  np.hstack(( Hgrd*0.0,      Vcavgrd.T*0.0,   Vmolgrd.T*0.0   )),
                                    np.hstack(( Vcavgrd*0.0,   Hcav*0.0,        Vmolcav.T*0.0   )),
                                    np.hstack(( Vmolgrd*0.0,   Vmolcav*0.0,     -1j*(self.Gamma/2)*Qmol )) ))

            self.Jt0 = np.vstack((  np.hstack(( Hgrd*0.0,      Vcavgrd.T*0.0,   Vmolgrd.T*0.0   )),
                                    np.hstack(( Vcavgrd*0.0,   Hcav*0.0,        Vmolcav.T*0.0   )),
                                    np.hstack(( Vmolgrd*0.0,   Vmolcav*0.0,     Jmol )) ))
            self.Icav = 1
            self.Imol = 2

        self.Jt = deepcopy(self.Jt0)
        self.Ht = deepcopy(self.Ht0)    

    def initialHamiltonian_Cavity_Radiation(self,Wgrd,Wcav,Wmol,Vndd,Vcav,Vrad,Wmax,damp,useQmatrix=False):
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
        
        # Construct the nearest dipole-dipole coupling
        for j in range(self.Nmol-1): 
            Hmol[j,   j+1] = Vndd
            Hmol[j+1, j  ] = Vndd
        Hmol[0,-1] = Vndd
        Hmol[-1,0] = Vndd

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

    def initialHamiltonian_Cavity_nonHermitian(self,Wgrd,Wcav,Wmol,Vndd,Vcav,Kcav,Gamma=0.0):
        """
        Construct the Hamiltonian in the form of 
        Ht0 = 
            | grd     | cav     | mol   
        grd | Hgrd    |         |       
        cav | Vcavgrd | Hcav    |         
        mol | Vmolgrd | Vmolcav | Hmol  
        """
        self.useQmatrix = True #Just to eliminate the rad part 
        self.Wmol = Wmol
        self.Gamma = Gamma

        Hgrd = np.eye(1) * Wgrd
        Hcav = np.eye(1) * Wcav
        Hmol = np.eye(self.Nmol) * Wmol
        Jmol = np.zeros((self.Nmol,self.Nmol),complex)

        Vmolgrd = np.ones((self.Nmol,1),complex)
        Vcavgrd = np.zeros((1,1),complex)
        Vmolcav = np.ones((self.Nmol,1),complex) * Vcav
        if not Kcav==0:
            for j in range(self.Nmol):
                # Vmolcav[j,0] = Vcav*np.sin(Kcav*np.pi*j/self.Nmol)
                Vmolcav[j,0] = Vcav*np.exp(-1j*(Kcav*np.pi*j/self.Nmol))
        
        # Construct the nearest dipole-dipole coupling
        for j in range(self.Nmol-1): 
            Hmol[j,   j+1] = Vndd
            Hmol[j+1, j  ] = Vndd
            Jmol[j,   j+1] = Vndd*1j
            Jmol[j+1, j  ] =-Vndd*1j
        Hmol[0,-1] = Vndd
        Hmol[-1,0] = Vndd
        Jmol[0,-1] =-Vndd*1j
        Jmol[-1,0] = Vndd*1j

        drive = 0.0
        Qmol = np.ones((self.Nmol,self.Nmol))
        self.Ht0 = np.vstack((  np.hstack(( Hgrd,          Vcavgrd.T,     Vmolgrd.T*drive   )),
                                np.hstack(( Vcavgrd,       Hcav,          np.conj(Vmolcav).T)),
                                np.hstack(( Vmolgrd*drive, Vmolcav,       Hmol -1j*(self.Gamma/2)*Qmol   )) ))
        
        self.Qmat = np.vstack((  np.hstack(( Hgrd*0.0,      Vcavgrd.T*0.0,   Vmolgrd.T*0.0   )),
                                np.hstack(( Vcavgrd*0.0,   Hcav*0.0,        Vmolcav.T*0.0   )),
                                np.hstack(( Vmolgrd*0.0,   Vmolcav*0.0,     -1j*(self.Gamma/2)*Qmol )) ))

        self.Jt0 = np.vstack((  np.hstack(( Hgrd*0.0,      Vcavgrd.T*0.0,   Vmolgrd.T*0.0   )),
                                np.hstack(( Vcavgrd*0.0,   Hcav*0.0,        Vmolcav.T*0.0   )),
                                np.hstack(( Vmolgrd*0.0,   Vmolcav*0.0,     Jmol )) ))
        self.Jt = deepcopy(self.Jt0)

        self.Ht = deepcopy(self.Ht0)        
        self.Icav = 1
        self.Imol = 2
 
    def initialHamiltonian_LossyCavity(self,Wgrd,Wcav,Wmol,Vndd,Vcav,Gamma_cav,Gamma_loc):
        """
        Construct the Hamiltonian in the form of 
        Ht0 = 
            | grd     | cav     | mol   
        grd | Hgrd    |         |       
        cav | Vcavgrd | Hcav    |         
        mol | Vmolgrd | Vmolcav | Hmol  
        """
        self.useQmatrix = True #Just to eliminate the rad part 
        self.Wmol = Wmol
        self.Gamma_cav = Gamma_cav
        self.Gamma_loc = Gamma_loc

        Hgrd = np.eye(1) * Wgrd
        Hcav = np.eye(1) * Wcav
        Hmol = np.eye(self.Nmol) * Wmol
        Cmol = np.eye(self.Nmol)
        Jmol = np.zeros((self.Nmol,self.Nmol),complex)

        Vmolgrd = np.ones((self.Nmol,1),complex)
        Vcavgrd = np.ones((1,1),complex)
        Vmolcav = np.ones((self.Nmol,1),complex) * Vcav
        
        # Construct the nearest dipole-dipole coupling
        for j in range(self.Nmol-1): 
            Hmol[j,   j+1] = Vndd
            Hmol[j+1, j  ] = Vndd
            Jmol[j,   j+1] = Vndd*1j
            Jmol[j+1, j  ] =-Vndd*1j
        Hmol[0,-1] = Vndd
        Hmol[-1,0] = Vndd
        Jmol[0,-1] =-Vndd*1j
        Jmol[-1,0] = Vndd*1j

        drive = 0.0

        self.Ht0 = np.vstack((  np.hstack(( Hgrd,           Vcavgrd.T*0.0,              Vmolgrd.T*0.0       )),
                                np.hstack(( Vcavgrd*0.0,    Hcav-1j*self.Gamma_cav,     np.conj(Vmolcav).T  )),
                                np.hstack(( Vmolgrd*0.0,    Vmolcav,                    Hmol-1j*self.Gamma_loc*Cmol   )) ))
        self.Hext= np.vstack((  np.hstack(( Hgrd*0.0,       Vcavgrd.T,               Vmolgrd.T*0.0           )),
                                np.hstack(( Vcavgrd*0.0,    Hcav*0.0,                np.conj(Vmolcav).T*0.0  )),
                                np.hstack(( Vmolgrd*0.0,    Vmolcav*0.0,             Hmol*0.0                )) ))
        self.Jt0 = np.vstack((  np.hstack(( Hgrd*0.0,      Vcavgrd.T*0.0,   Vmolgrd.T*0.0   )),
                                np.hstack(( Vcavgrd*0.0,   Hcav*0.0,        Vmolcav.T*0.0   )),
                                np.hstack(( Vmolgrd*0.0,   Vmolcav*0.0,     Jmol )) ))
        self.Jt = deepcopy(self.Jt0)

        self.Ht = deepcopy(self.Ht0)        
        self.Icav = 1
        self.Imol = 2

    def initialHamiltonian_LossyCavity_Radiation(self,Wgrd,Wcav,Wmol,Vndd,Vcav,Vrad,Wmax,damp,Gamma_loc,useQmatrix=False):
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
        self.Gamma_loc = Gamma_loc
        self.Erad = np.zeros(self.Nrad)

        Hgrd = np.eye(1) * Wgrd
        Hcav = np.eye(1) * Wcav
        Hmol = np.eye(self.Nmol) * Wmol
        Cmol = np.eye(self.Nmol)
        Hrad = np.zeros((self.Nrad,self.Nrad),complex)

        Vradmol = np.zeros((self.Nrad,self.Nmol),complex)
        Vmolgrd = np.ones((self.Nmol,1),complex)
        Vradgrd = np.zeros((self.Nrad,1),complex)
        Vcavgrd = np.ones((1,1),complex)
        Vmolcav = np.ones((self.Nmol,1),complex) * Vcav
        Vradcav = np.ones((self.Nrad,1),complex) * Vrad

        # Construct the molecule-radiation coupling
        Gamma_cav = 0.0
        for j in range(self.Nrad):
            self.Erad[j] = ( j - (self.Nrad-1)/2 ) * Wmax *2.0/(self.Nrad-1)
            Hrad[j,j] = self.Erad[j] - 1j*self.damp
            for i in range(self.Nmol):
                Vradmol[j,i] = Vrad # * (Wrad_width**2) / ( Erad[j]**2 + Wrad_width**2 )
            Gamma_cav += -2.0*1j*(Vradmol[j,0]**2)/Hrad[j,j] # set to be the same: 0
        #Gamma = 1j*Gamma*(Vrad**2)
        self.Gamma_cav = np.real(Gamma_cav)
        # print(self.Gamma_cav)

        # Construct the nearest dipole-dipole coupling
        for j in range(self.Nmol-1): 
            Hmol[j,   j+1] = Vndd
            Hmol[j+1, j  ] = Vndd
        Hmol[0,-1] = Vndd
        Hmol[-1,0] = Vndd

        self.Ht0 = np.vstack((  np.hstack(( Hgrd,           Vcavgrd.T*0.0,  Vmolgrd.T*0.0,                  Vradgrd.T*0.0   )),
                                np.hstack(( Vcavgrd*0.0,    Hcav,           Vmolcav.T,                      Vradcav.T       )),
                                np.hstack(( Vmolgrd*0.0,    Vmolcav,        Hmol-1j*self.Gamma_loc*Cmol,    Vradmol.T*0.0   )),
                                np.hstack(( Vradgrd*0.0,    Vradcav,        Vradmol*0.0,                    Hrad            )) ))
        self.Hext= np.vstack((  np.hstack(( Hgrd*0.0,       Vcavgrd.T,      Vmolgrd.T*0.0,   Vradgrd.T*0.0   )),
                                np.hstack(( Vcavgrd*0.0,    Hcav*0.0,       Vmolcav.T*0.0,  Vradcav.T*0.0   )),
                                np.hstack(( Vmolgrd*0.0,    Vmolcav*0.0,    Hmol*0.0,       Vradmol.T*0.0   )),
                                np.hstack(( Vradgrd*0.0,    Vradcav*0.0,    Vradmol*0.0,    Hrad*0.0        )) ))
        
        self.Ht = deepcopy(self.Ht0)
        
        self.Icav = 1
        self.Imol = 2
        self.Irad = self.Nmol+2

    def reset_Cgrd(self):
        self.Cj[0]=self.Cj[0]/np.abs(self.Cj[0])

    def updateDiagonalStaticDisorder(self,Delta):
        self.Ht = deepcopy(self.Ht0)

        self.Wstc = np.random.normal(0.0,Delta,self.Nmol) + self.Wmol
        for j in range(self.Nmol): 
            self.Ht[self.Imol+j,self.Imol+j] += self.Wstc[j]

    def updateCouplingStaticDisorder(self,Delta,Vcav,uniform_dist=False):
        self.Ht = deepcopy(self.Ht0)

        if uniform_dist:
            self.Theta_stc = np.random.uniform(0.0,Delta,self.Nmol)
            # self.Theta_stc = np.random.uniform(-Delta/2,Delta/2,self.Nmol)
            # print(np.average(np.cos(self.Theta_stc)))
            # print(np.var(np.cos(self.Theta_stc)))
        else:
            self.Theta_stc = np.random.normal(0.0,Delta,self.Nmol)
        for j in range(self.Nmol): 
            self.Ht[self.Icav,self.Imol+j] = Vcav*np.cos(self.Theta_stc[j])
            self.Ht[self.Imol+j,self.Icav] = Vcav*np.cos(self.Theta_stc[j])

    def updateCouplingDynamicDisorder(self,Delta,TauC,Vcav,dt,uniform_dist=False):
        # simulate Gaussian process
        # cf. George B. Rybicki's note
        # https://www.lanl.gov/DLDSTP/fast/OU_process.pdf
        self.Ht = deepcopy(self.Ht0)

        if not hasattr(self, 'Theta_dyn'):
            if uniform_dist:
                self.Theta_dyn = np.random.uniform(0.0,Delta,self.Nmol)
                # self.Theta_dyn = np.random.uniform(-Delta/2,Delta/2,self.Nmol)
            else:
                self.Theta_dyn = np.random.normal(0.0,Delta,self.Nmol)
        else:
            ri = np.exp(-dt/TauC) * (TauC>0.0)
            mean_it = ri*self.Theta_dyn
            sigma_it = Delta*np.sqrt(1.0-ri**2)
            self.Theta_dyn = np.random.normal(mean_it,sigma_it,self.Nmol)
        
        for j in range(self.Nmol): 
            self.Ht[self.Icav,self.Imol+j] = Vcav*np.cos(self.Theta_dyn[j])
            self.Ht[self.Imol+j,self.Icav] = Vcav*np.cos(self.Theta_dyn[j])

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

    def updateExternalDriving(self,DriveParam,time,dt):
        time0 = time
        time1 = time + dt/2
        time2 = time + dt
        if DriveParam['DriveType'] == 'None':
            drive0 = 0.0
        if DriveParam['DriveType'] == 'Constant':
            drive0 = DriveParam['DriveAmplitude']
        if DriveParam['DriveType'] == 'ContinuousSin':
            drive0 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time0)
            drive1 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time1)
            drive2 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time2)
            if time0-DriveParam['DrivePulseCenter']<0:
                drive0 = drive0*np.exp(-(time0-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time1-DriveParam['DrivePulseCenter']<0:
                drive1 = drive1*np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time2-DriveParam['DrivePulseCenter']<0:
                drive2 = drive2*np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
        if DriveParam['DriveType'] == 'ContinuousCos':
            drive0 = DriveParam['DriveAmplitude']*np.cos(DriveParam['DriveFrequency']*time0)
            drive1 = DriveParam['DriveAmplitude']*np.cos(DriveParam['DriveFrequency']*time1)
            drive2 = DriveParam['DriveAmplitude']*np.cos(DriveParam['DriveFrequency']*time2)
            if time0-DriveParam['DrivePulseCenter']<0:
                drive0 = drive0*np.exp(-(time0-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time1-DriveParam['DrivePulseCenter']<0:
                drive1 = drive1*np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time2-DriveParam['DrivePulseCenter']<0:
                drive2 = drive2*np.exp(-(time2-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
        if DriveParam['DriveType'] == 'ContinuousExp':
            drive0 = DriveParam['DriveAmplitude']*np.exp(1j*DriveParam['DriveFrequency']*time0)
            drive1 = DriveParam['DriveAmplitude']*np.exp(1j*DriveParam['DriveFrequency']*time1)
            drive2 = DriveParam['DriveAmplitude']*np.exp(1j*DriveParam['DriveFrequency']*time2)
            if time0-DriveParam['DrivePulseCenter']<0:
                drive0 = drive0*np.exp(-(time0-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time1-DriveParam['DrivePulseCenter']<0:
                drive1 = drive1*np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time2-DriveParam['DrivePulseCenter']<0:
                drive2 = drive2*np.exp(-(time2-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
        if DriveParam['DriveType'] == 'Pulse':
            drive0 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time0) * \
                    np.exp(-(time0-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            drive1 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time1) * \
                    np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            drive2 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time2) * \
                    np.exp(-(time2-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
        if DriveParam['DriveType'] == 'PulseCut':
            drive0 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time0) * \
                    np.exp(-(time0-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2) 
            drive1 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time1) * \
                    np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2) 
            drive2 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time2) * \
                    np.exp(-(time2-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2) 
            if time0-DriveParam['DrivePulseCenter'] >0:
                drive0 = 0.0
            if time1-DriveParam['DrivePulseCenter'] >0:
                drive1 = 0.0
            if time2-DriveParam['DrivePulseCenter'] >0:
                drive2 = 0.0
        
        self.Hext0 = self.Hext * drive0 + self.Hext.T * np.conj(drive0)
        self.Hext1 = self.Hext * drive1 + self.Hext.T * np.conj(drive1)
        self.Hext2 = self.Hext * drive2 + self.Hext.T * np.conj(drive2)
        
        return drive0
        
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
        """
        choose the initial Cj as a single exictation at the middle of the chain
        """
        self.Cj = np.zeros((self.Nmol,1),complex)
        self.Cj[int(self.Nmol/2)] = 1.0
        # j0 = int(self.Nmol/2)
        # width = 1
        # for j in range(self.Nmol):
        #     self.Cj[j,0] = np.exp(-(j-j0)**2/width**2/2)            
        # self.Cj = self.Cj/np.sqrt(np.sum(np.abs(self.Cj)**2))
        
        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    np.zeros((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Gaussian(self,width):
        """
        Initialize Cj as a Gaussian distribution centered at the middle of the chain
        """
        self.Cj = np.zeros((self.Nmol,1),complex)
        middle = int(self.Nmol/2)
        for j in range(self.Nmol):
            self.Cj[j] = np.exp(-(j-middle)**2/2/width**2)/np.sqrt(np.sqrt(np.pi)*width)
            # self.Cj[j] = np.exp(-(j-middle)**2/2/width**2)/np.sqrt(np.sqrt(np.pi)*width) * np.exp(1j*1.0*j) #WITH AN INITIAL MOMENTUM
        print(np.linalg.norm(self.Cj)**2)

        if hasattr(self, 'Icav'):
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    np.zeros((1,1),complex),  #cav
                                    self.Cj) )                #mol
        else:
            self.Cj = np.vstack( (np.zeros((1,1),complex),    #grd
                                    self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def initialCj_Eigenstate_Forward(self,Wmol,Vndd,initial_state=0):
        """
        Choose the initial state to be one of the eigenstate of the forward drift matrix
        """
        Amol = np.eye(self.Nmol) * Wmol
        for j in range(self.Nmol-1): 
            Amol[j,   j+1] = Vndd
            # Amol[j+1, j  ] = Vndd
        # Amol[0,-1] = Vndd
        Amol[-1,0] = Vndd
        W,U = np.linalg.eig(Amol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]

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

        return W

    def initialCj_Eigenstate(self,Vcav,initial_state=0):
        """
        Choose the initial state to be one of the eigenstate of Hmol 
        This function unifies initialCj_Eigenstate_Hmol and initialCj_Eigenstate_Hcavmol
        if Vcav==0: 
            Hmol
        else:
            Hcavmol
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        if Vcav==0.0:
            Hmol = self.Ht[self.Imol:self.Imol+self.Nmol,self.Imol:self.Imol+self.Nmol] ###NOTE: INCORRECT! THIS INCLUCE Qmat!!! 
            W,U = np.linalg.eigh(Hmol)
        else:
            Hcavmol = self.Ht[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]
            W,U = np.linalg.eigh(Hcavmol)
        
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]

        # Initialize state vector
        self.Cj = U.T[initial_state]
        self.Cj = self.Cj[..., None] 

        self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                              self.Cj) )                #mol
                              
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )
        self.Eall = W
        self.Evec = U

    def initialCj_Eigenstate_Hmol(self,initial_state=0):
        """
        Choose the initial state to be one of the eigenstate of Hmol
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hmol = self.Ht[self.Imol:self.Imol+self.Nmol,self.Imol:self.Imol+self.Nmol] ###NOTE: INCORRECT! THIS INCLUCE Qmat!!! 

        W,U = np.linalg.eigh(Hmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]

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
        self.Eall = W
        self.Evec = U

    def initialCj_Eigenstate_Hcavmol(self,initial_state):
        """
        Choose the initial state to be one of the eigenstate of Hmol+Hcav
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hcavmol = self.Ht[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]
        # Hcavmol = self.Ht0[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]

        W,U = np.linalg.eigh(Hcavmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]
        
        # Initialize state vector
        self.Cj = U.T[initial_state]
        self.Cj = self.Cj[..., None] 

        self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                              self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )
        self.Eall = W
        self.Evec = U

    def initialCall_Eigenstate(self,Vcav):
        """
        Set the initial state to be all the eigenstates 
        This function unifies initialCall_Eigenstate_Hmol and initialCall_Eigenstate_Hcavmol
        if Vcav==0: 
            Hmol
        else:
            Hcavmol
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        if Vcav==0: 
            Hmol = self.Ht[self.Imol:self.Imol+self.Nmol,self.Imol:self.Imol+self.Nmol] ###NOTE: INCORRECT! THIS INCLUCE Qmat!!! 
            W,U = np.linalg.eigh(Hmol)
        else:
            Hcavmol = self.Ht[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]
            W,U = np.linalg.eigh(Hcavmol)

        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]

        # Initialize the state matrix
        self.Eall = W
        self.Call = U
        self.Call = np.vstack( (np.zeros((1,len(U)),complex),  #grd
                                self.Call) )                #mol
        if not self.useQmatrix:
            self.Call = np.vstack( (self.Call,np.zeros((self.Nrad,len(U)),complex)) )

    def initialCall_Eigenstate_Hmol(self):
        """
        Choose the initial state to be one of the eigenstate of Hmol
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hmol = self.Ht[self.Imol:self.Imol+self.Nmol,self.Imol:self.Imol+self.Nmol] ###NOTE: INCORRECT! THIS INCLUCE Qmat!!! 

        W,U = np.linalg.eigh(Hmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]

        # Initialize state matrix
        self.Call = U
        self.Call = np.vstack( (np.zeros((1,len(U)),complex),  #grd
                                self.Call) )                #mol
        if not self.useQmatrix:
            self.Call = np.vstack( (self.Call,np.zeros((self.Nrad,len(U)),complex)) )
        self.Eall = W

    def initialCall_Eigenstate_Hcavmol(self):
        """
        Choose the initial state to be one of the eigenstate of Hmol+Hcav
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hcavmol = self.Ht[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]
        # Hcavmol = self.Ht0[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]

        W,U = np.linalg.eigh(Hcavmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]
        
        self.Call = U
        self.Call = np.vstack( (np.zeros((1,len(U)),complex),  #grd
                                self.Call) )                   #mol
        if not self.useQmatrix:
            self.Call = np.vstack( (self.Cj,np.zeros((self.Nrad,len(U)),complex)) )
        self.Eall = W

    def initialParition_Boltzmann(self,kBT):
        parition = np.sum(np.exp(-self.Eall/kBT))
        # append the ground state 
        self.Boltzmann = np.diag(np.exp(-self.Eall/kBT)/parition)

    def initialCj_Boltzman(self,hbar,kBT,most_prob=False):
        """
        Choose the initial state from the set of eigenfunctions based on Boltzman distribution exp(-E_n/kBT)
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hmol = self.Ht[self.Imol:self.Imol+self.Nmol,self.Imol:self.Imol+self.Nmol] ###NOTE: INCORRECT! THIS INCLUCE Qmat!!! 

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

    def initialCj_Polariton(self,initial_state):
        """
        Choose the initial state as the upper/lower polariton
        """
        # Use the updated Hamiltonian with this initial intermolecular coupling
        Hmol = (self.Ht-self.Qmat)[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]

        W,U = np.linalg.eigh(Hmol)
        #idx = W.argsort()[::-1]   
        idx = W.argsort()[:]
        W = W[idx]
        U = U[:,idx]
        
        # Initialize state vector
        self.Cj = U.T[initial_state]
        self.Cj = self.Cj[..., None] 

        self.Cj = np.vstack( (np.zeros((1,1),complex),  #grd
                              self.Cj) )                #mol
        if not self.useQmatrix:
            self.Cj = np.vstack( (self.Cj,np.zeros((self.Nrad,1),complex)) )

    def propagateCj_RK4(self,dt):
        ### RK4 propagation 
        if hasattr(self, 'Hext0'):
            K1 = -1j*np.dot(self.Ht+self.Hext0,self.Cj)
            K2 = -1j*np.dot(self.Ht+self.Hext1,self.Cj+dt*K1/2)
            K3 = -1j*np.dot(self.Ht+self.Hext1,self.Cj+dt*K2/2)
            K4 = -1j*np.dot(self.Ht+self.Hext2,self.Cj+dt*K3)
        else:
            K1 = -1j*np.dot(self.Ht,self.Cj)
            K2 = -1j*np.dot(self.Ht,self.Cj+dt*K1/2)
            K3 = -1j*np.dot(self.Ht,self.Cj+dt*K2/2)
            K4 = -1j*np.dot(self.Ht,self.Cj+dt*K3)
        self.Cj += (K1+2*K2+2*K3+K4)*dt/6

    def propagateCall_RK4(self,dt):
        ### RK4 propagation 
        K1 = -1j*np.dot(self.Ht,self.Call)
        K2 = -1j*np.dot(self.Ht,self.Call+dt*K1/2)
        K3 = -1j*np.dot(self.Ht,self.Call+dt*K2/2)
        K4 = -1j*np.dot(self.Ht,self.Call+dt*K3)
        self.Call += (K1+2*K2+2*K3+K4)*dt/6
        
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

        self.Xj = np.random.normal(0.0, np.sqrt(self.kBT/self.Kconst), self.Nmol)
        self.Vj = np.random.normal(0.0, np.sqrt(self.kBT/self.mass),   self.Nmol)
        
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

    def getOrderParameter(self):
        order = 0.0*1j
        for i in range(self.Imol,self.Imol+self.Nmol):
            order += self.Cj[i]/np.abs(self.Cj[i])/self.Nmol
        # return np.abs(order), np.angle(order)
        return np.abs(order), np.imag(np.log(order/np.abs(order)))

    def getPopulation_ground(self):
        return np.linalg.norm(self.Cj[0])**2
    
    def getPopulation_system(self):
        return np.linalg.norm(self.Cj[self.Imol:self.Imol+self.Nmol])**2

    def getPopulation_radiation(self):
        if not self.useQmatrix:
            return np.linalg.norm(self.Cj[self.Irad:])**2
        else:
            return 0.0
            
    def getPopulation_cavity(self):
        if hasattr(self, 'Icav'):
            return np.linalg.norm(self.Cj[self.Icav])**2
        else:
            return 0.0

    def getPopulation_polariton(self):
        ### note: these UP and LP states are only for the on-resonance case. 
        bright = np.ones((self.Nmol,1),complex)/np.sqrt(self.Nmol)/np.sqrt(2.0)
        if hasattr(self, 'Icav'):
            upper = np.vstack( (np.zeros((1,1),complex),    #grd
                                np.ones((1,1),complex)/np.sqrt(2.0),  #cav
                                bright) )                #mol
            lower = np.vstack( (np.zeros((1,1),complex),    #grd
                                -np.ones((1,1),complex)/np.sqrt(2.0),  #cav
                                bright) )                #mol
        P_upper = np.abs(np.dot(upper.T,self.Cj))**2
        P_lower = np.abs(np.dot(lower.T,self.Cj))**2


        return P_upper[0,0],P_lower[0,0]
    
    def getPopulation_bright(self):
        bright = np.ones((self.Nmol,1),complex)/np.sqrt(self.Nmol)
        if hasattr(self, 'Icav'):
            bright = np.vstack( (np.zeros((1,1),complex),    #grd
                                    np.zeros((1,1),complex),  #cav
                                    bright) )                #mol
        else:
            bright = np.vstack( (np.zeros((1,1),complex),    #grd
                                    bright) )                #mol
        P_bright = np.abs(np.dot(bright.T,self.Cj))**2
        P_dark = np.linalg.norm(self.Cj[self.Imol:self.Imol+self.Nmol])**2 - P_bright

        return P_bright[0,0],P_dark[0,0]
    
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

    def getCurrentCorrelation(self):
        if hasattr(self, 'J0Cj'):
            # self.Jt = deepcopy(self.Ht)
            self.Jt = np.zeros_like(self.Ht)
            for j in range(self.Nmol-1): 
                self.Jt[self.Imol+j,   self.Imol+j+1] = self.Ht[self.Imol+j,   self.Imol+j+1]*1j
                self.Jt[self.Imol+j+1, self.Imol+j]   =-self.Ht[self.Imol+j+1, self.Imol+j]*1j   
            
            self.Jt[self.Imol,self.Imol+self.Nmol-1] =-self.Ht[self.Imol,self.Imol+self.Nmol-1]*1j
            self.Jt[self.Imol+self.Nmol-1,self.Imol] = self.Ht[self.Imol+self.Nmol-1,self.Imol]*1j 
            # Here Cj is at time t
            # self.JtCj = np.dot(self.Jt,self.Cj)
        else: #first step only 
            self.J0Cj = np.dot(self.Jt0,self.Cj)
            self.Jt = deepcopy(self.Jt0)

        CJJ = np.dot(np.conj(self.Cj).T,np.dot(self.Jt,self.J0Cj))
        Javg = np.dot(np.conj(self.Cj).T,np.dot(self.Jt,self.Cj))
        return Javg[0,0], CJJ[0,0]

    def getCurrentCorrelation_all(self):
        if hasattr(self, 'J0Call'):
            # self.Jt = deepcopy(self.Ht)
            self.Jt = np.zeros_like(self.Ht)
            for j in range(self.Nmol-1): 
                self.Jt[self.Imol+j,   self.Imol+j+1] = self.Ht[self.Imol+j,   self.Imol+j+1]*1j
                self.Jt[self.Imol+j+1, self.Imol+j]   =-self.Ht[self.Imol+j+1, self.Imol+j]*1j   
            
            self.Jt[self.Imol,self.Imol+self.Nmol-1] =-self.Ht[self.Imol,self.Imol+self.Nmol-1]*1j
            self.Jt[self.Imol+self.Nmol-1,self.Imol] = self.Ht[self.Imol+self.Nmol-1,self.Imol]*1j 
            # Here Cj is at time t
            # self.JtCj = np.dot(self.Jt,self.Cj)
        else: #first step only 
            self.J0Call = np.dot(self.Jt0,self.Call)
            self.Jt = deepcopy(self.Jt0)

        CJJ = np.dot(np.conj(self.Call).T,np.dot(self.Jt,self.J0Call))
        J = np.dot(np.conj(self.Call).T,np.dot(self.Jt,self.Call))

        CJJall = np.dot(np.dot(np.conj(self.Call).T,np.dot(self.Jt,self.J0Call)),self.Boltzmann)
        Jall = np.dot(np.dot(np.conj(self.Call).T,np.dot(self.Jt,self.Call)),self.Boltzmann)
        
        CJJavg = np.trace(CJJall)
        Javg = np.trace(Jall)
        return Javg, CJJavg

    def getCurrentCorrelation_stationary(self,dt,Ntimes,kBT,Vcav):
        times = np.arange(Ntimes)*dt
        if Vcav==0.0:
            Nsize = self.Nmol
            Jmol = self.Jt0[self.Imol:self.Imol+self.Nmol,self.Imol:self.Imol+self.Nmol]
        else:
            Nsize = self.Nmol +1 
            Jmol = self.Jt0[self.Icav:self.Imol+self.Nmol,self.Icav:self.Imol+self.Nmol]
        J0mn = np.dot(np.conj(self.Evec).T,np.dot(Jmol,self.Evec))
    
        CJJ_avg = np.zeros(Ntimes,complex)
        Partition = 0.0
        for m in range(Nsize):
            CJJ_ana = np.zeros(Ntimes,complex)
            for n in range(Nsize):
                CJJ_ana = CJJ_ana + np.exp(1j*(self.Eall[m]-self.Eall[n])*times)*np.abs(J0mn[m,n])**2
            # print(CJJ_ana[-1])
            CJJ_avg = CJJ_avg + CJJ_ana * np.exp(-self.Eall[m]/kBT)
            Partition = Partition + np.exp(-self.Eall[m]/kBT)
        CJJ_avg = CJJ_avg/Partition

        return CJJ_avg

    def propagateJ0Cj_RK4(self,dt):
        ### RK4 propagation 
        K1 = -1j*np.dot(self.Ht,self.J0Cj)
        K2 = -1j*np.dot(self.Ht,self.J0Cj+dt*K1/2)
        K3 = -1j*np.dot(self.Ht,self.J0Cj+dt*K2/2)
        K4 = -1j*np.dot(self.Ht,self.J0Cj+dt*K3)
        self.J0Cj += (K1+2*K2+2*K3+K4)*dt/6

    def propagateJ0Call_RK4(self,dt):
        ### RK4 propagation 
        K1 = -1j*np.dot(self.Ht,self.J0Call)
        K2 = -1j*np.dot(self.Ht,self.J0Call+dt*K1/2)
        K3 = -1j*np.dot(self.Ht,self.J0Call+dt*K2/2)
        K4 = -1j*np.dot(self.Ht,self.J0Call+dt*K3)
        self.J0Call += (K1+2*K2+2*K3+K4)*dt/6

class SingleExcitationWithThreeStates():

    def __init__(self,seed=None):
        np.random.seed(seed)

    def initialHamiltonian_FourStates(self,Wmol,Wcav,Vgrdchs,Vchsloc,Vchscav,Vloccav,Gchs,Gloc,Gcav):
        """
        Construct the Hamiltonian in the form of 
        Ht0 = 
                | grd       | chs       | loc       | grd+cav
        grd     | Hgrd      | Vgrdchs   |           | Vgrdcav
        chs     |           | Hchs      | Vchsloc   | Vchscav
        loc     |           |           | Hloc      | Vloccav
        grd+cav |           |           |           | Hcav
        """
        self.useQmatrix = True #Just to eliminate the rad part 

        self.Ht0 = np.zeros((4,4),complex)
        self.Ht0[0,0] = Wmol[0]
        self.Ht0[1,1] = Wmol[1] - 1j*Gchs
        self.Ht0[2,2] = Wmol[2] - 1j*Gloc
        self.Ht0[3,3] = Wmol[0]+Wcav - 1j*Gcav

        self.Ht0[0,1] = Vgrdchs; self.Ht0[1,0] = Vgrdchs
        self.Ht0[1,2] = Vchsloc; self.Ht0[2,1] = Vchsloc
        self.Ht0[1,3] = Vchscav; self.Ht0[3,1] = Vchscav
        self.Ht0[2,3] = Vloccav; self.Ht0[3,2] = Vloccav
        
        drive = 0.0

        self.Hext= np.zeros((4,4),complex)
        self.Hext[0,3] = 1.0; self.Hext[3,0] = 1.0

        self.Ht = deepcopy(self.Ht0)

        self.Icav = 3
        self.Imol = 0

    def reset_Cgrd(self):
        self.Cj[0]=self.Cj[0]/np.abs(self.Cj[0])

    def updateExternalDriving(self,DriveParam,time,dt):
        time0 = time
        time1 = time + dt/2
        time2 = time + dt
        if DriveParam['DriveType'] == 'None':
            drive0 = 0.0
        if DriveParam['DriveType'] == 'Constant':
            drive0 = DriveParam['DriveAmplitude']
        if DriveParam['DriveType'] == 'ContinuousSin':
            drive0 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time0)
            drive1 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time1)
            drive2 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time2)
            if time0-DriveParam['DrivePulseCenter']<0:
                drive0 = drive0*np.exp(-(time0-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time1-DriveParam['DrivePulseCenter']<0:
                drive1 = drive1*np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time2-DriveParam['DrivePulseCenter']<0:
                drive2 = drive2*np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
        if DriveParam['DriveType'] == 'ContinuousCos':
            drive0 = DriveParam['DriveAmplitude']*np.cos(DriveParam['DriveFrequency']*time0)
            drive1 = DriveParam['DriveAmplitude']*np.cos(DriveParam['DriveFrequency']*time1)
            drive2 = DriveParam['DriveAmplitude']*np.cos(DriveParam['DriveFrequency']*time2)
            if time0-DriveParam['DrivePulseCenter']<0:
                drive0 = drive0*np.exp(-(time0-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time1-DriveParam['DrivePulseCenter']<0:
                drive1 = drive1*np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time2-DriveParam['DrivePulseCenter']<0:
                drive2 = drive2*np.exp(-(time2-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
        if DriveParam['DriveType'] == 'ContinuousExp':
            drive0 = DriveParam['DriveAmplitude']*np.exp(1j*DriveParam['DriveFrequency']*time0)
            drive1 = DriveParam['DriveAmplitude']*np.exp(1j*DriveParam['DriveFrequency']*time1)
            drive2 = DriveParam['DriveAmplitude']*np.exp(1j*DriveParam['DriveFrequency']*time2)
            if time0-DriveParam['DrivePulseCenter']<0:
                drive0 = drive0*np.exp(-(time0-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time1-DriveParam['DrivePulseCenter']<0:
                drive1 = drive1*np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            if time2-DriveParam['DrivePulseCenter']<0:
                drive2 = drive2*np.exp(-(time2-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
        if DriveParam['DriveType'] == 'Pulse':
            drive0 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time0) * \
                    np.exp(-(time0-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            drive1 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time1) * \
                    np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
            drive2 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time2) * \
                    np.exp(-(time2-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2)
        if DriveParam['DriveType'] == 'PulseCut':
            drive0 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time0) * \
                    np.exp(-(time0-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2) 
            drive1 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time1) * \
                    np.exp(-(time1-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2) 
            drive2 = DriveParam['DriveAmplitude']*np.sin(DriveParam['DriveFrequency']*time2) * \
                    np.exp(-(time2-DriveParam['DrivePulseCenter'])**2/DriveParam['DrivePulseWidth']**2) 
            if time0-DriveParam['DrivePulseCenter'] >0:
                drive0 = 0.0
            if time1-DriveParam['DrivePulseCenter'] >0:
                drive1 = 0.0
            if time2-DriveParam['DrivePulseCenter'] >0:
                drive2 = 0.0
        
        self.Hext0 = self.Hext * drive0 + self.Hext.T * np.conj(drive0)
        self.Hext1 = self.Hext * drive1 + self.Hext.T * np.conj(drive1)
        self.Hext2 = self.Hext * drive2 + self.Hext.T * np.conj(drive2)
        
        return drive0

    def initialCj_Ground(self):
        self.Cj = np.zeros((4,1),complex)
        self.Cj[0,0] = 1.0

    def propagateCj_RK4(self,dt):
        ### RK4 propagation 
        if hasattr(self, 'Hext0'):
            K1 = -1j*np.dot(self.Ht+self.Hext0,self.Cj)
            K2 = -1j*np.dot(self.Ht+self.Hext1,self.Cj+dt*K1/2)
            K3 = -1j*np.dot(self.Ht+self.Hext1,self.Cj+dt*K2/2)
            K4 = -1j*np.dot(self.Ht+self.Hext2,self.Cj+dt*K3)
        else:
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

    def getPopulation_ground(self):
        return np.linalg.norm(self.Cj[0])**2

    def getPopulation_cavity(self):
        if hasattr(self, 'Icav'):
            return np.abs(self.Cj[self.Icav])**2
        else:
            return 0.0

