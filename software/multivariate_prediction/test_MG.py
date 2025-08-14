import os
import numpy as np
from benchmark import Henon

class Mask:
    def __init__(self, N_u, N_x, gamma, seed=0):
        np.random.seed(seed)
        self.Win = gamma*(-1)**np.random.randint(0,2,(N_u,N_x))
        
    def __call__(self, u):    
        return np.dot(u,self.Win)

class DFR:
    def __init__(self, N_x, eta, p=1, theta=0.2):
        self.N_x = N_x
        self.eta = eta
        self.p = p
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
    def __init__(self, N_r, N_y):
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

if __name__ == "__main__":   
    dataset = 'Chen_x'

    if(not os.path.isfile(f'./result/{dataset}/dataset.npz')):
        try:
            os.mkdir('./result')
        except:
            pass
        for attractor in ['Henon']:
            for xyz in ['x','y']:
                try:
                    os.mkdir(f'./result/{attractor}_{xyz}')
                except:
                    pass
                dynamics = Henon()
                u, d, T, eval_func = dynamics.generate_data(xyz)
                np.savez(f'./result/{attractor}_{xyz}/dataset', u=u, d=d, T=T, eval_func=eval_func)

    data = np.load(f'./result/{dataset}/dataset.npz')
    u = data['u']
    d = data['d']
    T_skip,T_train,T_test = data['T']
    eval_func = data['eval_func']
    k = 1/np.max(u)
    train_U = u[:T_skip+T_train]*k
    test_U = u[T_skip+T_train:]*k
    train_D = d[:T_skip+T_train]
    test_D = d[T_skip+T_train:]
    
    N_u = u.shape[1]
    N_x = 100
    N_y = 1
    
    best_RMSE = 1e+16
    best_param = []
    
    # for gamma in [1e-5,1e-4,1e-3,1e-2,1e-1]:
    #     for eta in [1e-5,1e-4,1e-3,1e-2,1e-1]:
    #         for beta in [1e-20,1e-16,1e-12,1e-8,1e-4]:
    for gamma in [0.02,0.03,0.04]:
        for eta in [0.2,0.3,0.4]:
            for beta in [1e-13,1e-12,1e-11]:
                mask = Mask(N_u,N_x,gamma)
                reservoir = DFR(N_x,eta)
                output = Output(N_x+1,N_y)
                optimizer = Tikhonov(N_x+1, N_y, beta)
                for i in range(T_skip+T_train):
                    X = reservoir(mask(train_U[i]))
                    if(i>=T_skip):
                        optimizer(train_D[i], np.append(1,X))
                output.setweight(optimizer.get_Wout_opt())
                
                Y_pred = []
                for i in range(T_test):
                    X = reservoir(mask(test_U[i]))
                    y_pred = output(np.append(1,X))
                    Y_pred.append(y_pred)
                test_Y = np.array(Y_pred).reshape(-1)

                test_RMSE = np.sqrt(((test_Y-test_D)**2).mean())
                
                if(test_RMSE < best_RMSE):
                    best_RMSE = test_RMSE
                    best_param = [gamma,eta,beta]
    
    print(best_RMSE,best_param)