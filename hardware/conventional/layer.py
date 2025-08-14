import numpy as np

seed = 0

class Mask:
    def __init__(self, N_u, N_x, gamma, seed=seed):
        np.random.seed(seed)
        self.Win = gamma*0.1*(-1)**np.random.randint(0,2,(N_x,N_u))
        
    def __call__(self, u):
        return np.dot(self.Win, u)

class DFR:
    def __init__(self, eta, p, N_x, theta):
        self.eta = eta
        self.p = p
        self.N_x = N_x
        self.theta = theta
        self.x_prev = np.zeros(N_x)
        self.exp = np.exp(-self.theta)
    
    def g(self, x, j):
        return self.eta*(x+j)/(1+(x+j)**self.p)
    
    def MG(self, x_prevnode, x_prevstep, j):
        return x_prevnode*self.exp \
            + (1-self.exp)*self.g(x_prevstep,j)
        
    def __call__(self, J):
        X = np.zeros(self.N_x)
        X[0] = self.MG(self.x_prev[self.N_x-1], self.x_prev[0], J[0])
        for i in range(self.N_x-1):
            X[i+1] = self.MG(X[i], self.x_prev[i+1], J[i+1])
        self.x_prev = X
        return self.x_prev
    
    def refresh(self):
        self.x_prev = np.zeros(self.N_x)
    
class Output:
    def __init__(self, N_r, N_y, seed=seed):
        np.random.seed(seed=seed)
        self.Wout = np.random.normal(size=(N_y, N_r))
    
    def __call__(self, r):
        return np.dot(self.Wout, r)
    
    def setweight(self, Wout_opt):
        self.Wout = Wout_opt
        
class Tikhonov:
    def __init__(self, N_r, N_y, beta):
        self.beta = beta
        self.R_RT = np.zeros((N_r, N_r))
        self.D_RT = np.zeros((N_y, N_r))
        self.I = np.identity(N_r)
        
    def __call__(self, d, r):
        r = np.reshape(r, (-1, 1)).astype(np.float64)
        d = np.reshape(d, (-1, 1))
        self.R_RT += np.dot(r, r.T)
        self.D_RT += np.dot(d, r.T)
    
    def get_Wout_opt(self):
        R_pseudo_inv = np.linalg.inv(self.R_RT + self.beta*self.I)
        Wout_opt = np.dot(self.D_RT, R_pseudo_inv)
        return Wout_opt