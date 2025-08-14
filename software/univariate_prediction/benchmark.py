import numpy as np
import matplotlib.pyplot as plt

class NARMA:
    def __init__(self, m=10, a1=0.3, a2=0.05, a3=1.5, a4=0.1):
        self.m = m
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.eval_func = 'NRMSE'

    def generate_data(self, T_skip=100, T_train=5900, T_test=4000):
        n = self.m
        y = [0]*n
        np.random.seed(seed=0)
        u = np.random.uniform(0, 0.5, T_skip+T_train+T_test)
        while n < T_skip+T_train+T_test:
            y_n = self.a1*y[n-1] + self.a2*y[n-1]*(np.sum(y[n-self.m:n-1])) \
                + self.a3*u[n-self.m]*u[n] + self.a4
            y.append(y_n)
            n += 1
        return u, np.array(y), [T_skip,T_train,T_test], self.eval_func

class Spectrum:
    def __init__(self, noise, ant):
        # noise: 10, 15, 20
        # ant: 2, 4, 6
        self.noise = noise
        self.ant = ant
        self.T_skip = 20
        self.T_train = 980
        self.T_test = 5102 # 5082
        self.eval_func = 'AUC'

    def generate_data(self):
        spectrum_vector = np.genfromtxt(f"./dataset/spectrum/spectrum_-{self.noise}_db_{self.ant}_ant.csv", delimiter=",")
        u = spectrum_vector[:,0]
        d = spectrum_vector[:,1]
        return u, d, [self.T_skip,self.T_train,self.T_test], self.eval_func

class MackeyGlass:
    def __init__(self, beta=0.2, gamma=0.1, n=10, tau=17, init=1.2):
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau
        self.init = init
        self.eval_func = 'RMSE'
    
    def f(self,y,y_delay):
        return self.beta*y_delay/(1+y_delay**self.n)-self.gamma*y

    def generate_data(self, T_skip=100, T_train=400, T_test=500, step=10):
        n = self.tau*step+1
        u = np.ones(n)*self.init
        T = T_skip+T_train+T_test
        while n < (T+1)*step:
            k_1 = self.f(u[n-1],u[n-self.tau*step-1])
            k_2 = self.f(u[n-1]+k_1/(2*step),u[int(n-self.tau*step-1+step/2)])
            k_3 = self.f(u[n-1]+k_2/(2*step),u[int(n-self.tau*step-1+step/2)])
            k_4 = self.f(u[n-1]+k_3/step,u[n-self.tau*(step-1)-1])
            u = np.append(u,u[n-1]+1/(6*step)*(k_1+2*k_2+2*k_3+k_4))
            n += 1
        
        y = u[step:]
        u = u[0:T*step]
        
        return u[::step], y[::step], [T_skip,T_train,T_test], self.eval_func

class DST:
    def __init__(self, year):
        self.year = year
        self.T_skip = 100
        self.T_train = 2900
        self.T_test = 1000
        self.eval_func = 'RMSE'

    def generate_data(self):
        f_in = open(f"./dataset/DST/DST{self.year}.txt", mode='r')
        f_out = open(f"./dataset/DST/DST{self.year}_modified.txt", mode='w')
        for in_line in f_in:
            f_out.write(','.join(in_line[20:116].replace('-',' -').split()))
            f_out.write('\n')
        f_in.close()
        f_out.close()
        data = np.loadtxt(f"./dataset/DST/DST{self.year}_modified.txt", delimiter=",")
        data = data.reshape(-1)
        T = self.T_skip+self.T_train+self.T_test
        u = data[len(data)-1-T:len(data)-1]
        d = data[len(data)-T:]
        return u, d, [self.T_skip,self.T_train,self.T_test], self.eval_func