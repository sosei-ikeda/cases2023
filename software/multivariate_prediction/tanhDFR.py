import os
import numpy as np
from benchmark import Lorenz, Rossler

class Mask:
    def __init__(self, N_u, N_x, seed=0):
        np.random.seed(seed)
        self.Win = (-1)**np.random.randint(0,2,(N_u,N_x))
        
    def __call__(self, u):    
        return np.dot(u,self.Win)

class DFR:
    def __init__(self, N_x, alpha, A, B=0.82):
        self.N_x = N_x
        self.A = A
        self.B = B
        self.alpha = alpha
        self.x_prev = np.zeros(N_x)
    
    def g(self, x, j):
        # return self.alpha*np.tanh((x+j)/self.alpha)
        return self.alpha*((x+j)/self.alpha - 1/3*((x+j)/self.alpha)**3 + 2/15*((x+j)/self.alpha)**5) 
    
    def MG(self, x_prevnode, x_prevstep, j):
        return x_prevnode*self.B + self.A*self.g(x_prevstep,j)
        
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
    for i in range(8):
        if(i==0):
            dataset = 'Lorenz_x'
            alpha = 10
            A = 0.1
            beta = 1e-5
        elif(i==1):
            dataset = 'Lorenz_y'
            alpha = 4
            A = 0.15
            beta = 1e-4
        elif(i==2):
            dataset = 'Lorenz_z'
            alpha = 15 
            A = 0.07
            beta = 1e-6
        elif(i==3):
            dataset = 'Rossler_x'
            alpha = 4000
            A = 0.04
            beta = 1e-4
        elif(i==4):
            dataset = 'Rossler_y'
            alpha = 400
            A = 0.02
            beta = 1e-5
        elif(i==5):
            dataset = 'Rossler_z'
            alpha = 10
            A = 0.1
            beta = 1e-6
        elif(i==6):
            dataset = 'Chen_x'
            alpha = 7
            A = 0.07
            beta = 1e-5
        else:
            dataset = 'Henon_x'
            alpha = 6
            A = 0.25
            beta = 1e-7
        
        print(dataset)
    
        if(not os.path.isfile(f'./result/{dataset}/dataset.npz')):
            try:
                os.mkdir('./result')
            except:
                pass
            for series in ['Lorenz', 'Rossler']:
                for xyz in ['x','y','z']:
                    try:
                        os.mkdir(f'./result/{series}_{xyz}')
                    except:
                        pass
                    if(series=='Lorenz'):
                        dynamics = Lorenz()
                    else:
                        dynamics = Rossler()
                    u, d, T, eval_func = dynamics.generate_data(xyz)
                    np.savez(f'./result/{series}_{xyz}/dataset', u=u, d=d, T=T, eval_func=eval_func)
    
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
        
        mask = Mask(N_u,N_x)
        reservoir = DFR(N_x,alpha,A)
        output = Output(N_x+1,N_y)
        optimizer = Tikhonov(N_x+1, N_y, beta*k**2)
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
        print('test_RMSE =', test_RMSE)
        test_MAE = np.abs(test_Y-test_D).mean()
        print('test_MAE =', test_MAE)
        test_MAPE = np.abs((test_Y-test_D)/test_Y).mean()
        print('test_MAPE =', test_MAPE)
        test_MSE = ((test_Y-test_D)**2).mean()
        print('test_MSE =', test_MSE, '\n')