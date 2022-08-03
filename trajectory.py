import numpy as np

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