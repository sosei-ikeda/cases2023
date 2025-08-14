import numpy as np
import matplotlib.pyplot as plt

class Lorenz:
    def __init__(self, sigma=10, beta=8/3, rho=28, init=[1,0,1]):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.init = init
        self.eval_func = 'RMSE'
    
    def f_x(self,x,y,z):
        return self.sigma*(y-x)
    def f_y(self,x,y,z):
        return x*(self.rho-z)-y
    def f_z(self,x,y,z):
        return x*y-self.beta*z
    
    def Runge_Kutta(self,x,y,z,step):
        k_x_1 = self.f_x(x,y,z)
        k_y_1 = self.f_y(x,y,z)
        k_z_1 = self.f_z(x,y,z)
        
        k_x_2 = self.f_x(x+1/(2*step)*k_x_1,y+1/(2*step)*k_y_1,z+1/(2*step)*k_z_1)
        k_y_2 = self.f_y(x+1/(2*step)*k_x_1,y+1/(2*step)*k_y_1,z+1/(2*step)*k_z_1)
        k_z_2 = self.f_z(x+1/(2*step)*k_x_1,y+1/(2*step)*k_y_1,z+1/(2*step)*k_z_1)
        
        k_x_3 = self.f_x(x+1/(2*step)*k_x_2,y+1/(2*step)*k_y_2,z+1/(2*step)*k_z_2)
        k_y_3 = self.f_y(x+1/(2*step)*k_x_2,y+1/(2*step)*k_y_2,z+1/(2*step)*k_z_2)
        k_z_3 = self.f_z(x+1/(2*step)*k_x_2,y+1/(2*step)*k_y_2,z+1/(2*step)*k_z_2)
        
        k_x_4 = self.f_x(x+1/step*k_x_3,y+1/step*k_y_3,z+1/step*k_z_3)
        k_y_4 = self.f_y(x+1/step*k_x_3,y+1/step*k_y_3,z+1/step*k_z_3)
        k_z_4 = self.f_z(x+1/step*k_x_3,y+1/step*k_y_3,z+1/step*k_z_3)
        
        x_next = x+1/(6*step)*(k_x_1+2*k_x_2+2*k_x_3+k_x_4)
        y_next = y+1/(6*step)*(k_y_1+2*k_y_2+2*k_y_3+k_y_4)
        z_next = z+1/(6*step)*(k_z_1+2*k_z_2+2*k_z_3+k_z_4)
        
        return x_next, y_next, z_next

    def generate_data(self, xyz, T_skip=100, T_train=8900, T_test=3000, step=100):
        T = T_skip+T_train+T_test
        
        x = np.array([self.init[0]])
        y = np.array([self.init[1]])
        z = np.array([self.init[2]])
        for i in range(T):
            x_next,y_next,z_next = self.Runge_Kutta(x[i],y[i],z[i],step)
            x = np.append(x,x_next)
            y = np.append(y,y_next)
            z = np.append(z,z_next)
            
        x_plus = x[1:]
        y_plus = y[1:]
        z_plus = z[1:]
        
        u = x[0:T]
        u = np.vstack((u, y[0:T]))
        u = np.vstack((u, z[0:T]))
        
        if(xyz=='x'):
            u_plus = x_plus
        elif(xyz=='y'):
            u_plus = y_plus
        elif(xyz=='z'):
            u_plus = z_plus
        else:
            raise ValueError
        
        return u.T, u_plus, [T_skip,T_train,T_test], self.eval_func

class Rossler:
    def __init__(self, a=0.15, b=0.2, c=10, init=[1,0,1]):
        self.a = a
        self.b = b
        self.c = c
        self.init = init
        self.eval_func = 'RMSE'
    
    def f_x(self,x,y,z):
        return -y-z
    def f_y(self,x,y,z):
        return x+self.a*y
    def f_z(self,x,y,z):
        return self.b+z*(x-self.c)
    
    def Runge_Kutta(self,x,y,z,step):
        k_x_1 = self.f_x(x,y,z)
        k_y_1 = self.f_y(x,y,z)
        k_z_1 = self.f_z(x,y,z)
        
        k_x_2 = self.f_x(x+1/(2*step)*k_x_1,y+1/(2*step)*k_y_1,z+1/(2*step)*k_z_1)
        k_y_2 = self.f_y(x+1/(2*step)*k_x_1,y+1/(2*step)*k_y_1,z+1/(2*step)*k_z_1)
        k_z_2 = self.f_z(x+1/(2*step)*k_x_1,y+1/(2*step)*k_y_1,z+1/(2*step)*k_z_1)
        
        k_x_3 = self.f_x(x+1/(2*step)*k_x_2,y+1/(2*step)*k_y_2,z+1/(2*step)*k_z_2)
        k_y_3 = self.f_y(x+1/(2*step)*k_x_2,y+1/(2*step)*k_y_2,z+1/(2*step)*k_z_2)
        k_z_3 = self.f_z(x+1/(2*step)*k_x_2,y+1/(2*step)*k_y_2,z+1/(2*step)*k_z_2)
        
        k_x_4 = self.f_x(x+1/step*k_x_3,y+1/step*k_y_3,z+1/step*k_z_3)
        k_y_4 = self.f_y(x+1/step*k_x_3,y+1/step*k_y_3,z+1/step*k_z_3)
        k_z_4 = self.f_z(x+1/step*k_x_3,y+1/step*k_y_3,z+1/step*k_z_3)
        
        x_next = x+1/(6*step)*(k_x_1+2*k_x_2+2*k_x_3+k_x_4)
        y_next = y+1/(6*step)*(k_y_1+2*k_y_2+2*k_y_3+k_y_4)
        z_next = z+1/(6*step)*(k_z_1+2*k_z_2+2*k_z_3+k_z_4)
        
        return x_next, y_next, z_next

    def generate_data(self, xyz, T_skip=100, T_train=8900, T_test=3000, step=100):
        T = T_skip+T_train+T_test
        
        x = np.array([self.init[0]])
        y = np.array([self.init[1]])
        z = np.array([self.init[2]])
        for i in range(T):
            x_next,y_next,z_next = self.Runge_Kutta(x[i],y[i],z[i],step)
            x = np.append(x,x_next)
            y = np.append(y,y_next)
            z = np.append(z,z_next)
            
        x_plus = x[1:]
        y_plus = y[1:]
        z_plus = z[1:]
        
        u = x[0:T]
        u = np.vstack((u, y[0:T]))
        u = np.vstack((u, z[0:T]))
            
        if(xyz=='x'):
            u_plus = x_plus
        elif(xyz=='y'):
            u_plus = y_plus
        elif(xyz=='z'):
            u_plus = z_plus
        else:
            raise ValueError
        
        return u.T, u_plus, [T_skip,T_train,T_test], self.eval_func

class Chen:
    def __init__(self, a=40, b=3, c=28, init=[-0.1,0.5,-0.6]):
        self.a = a
        self.b = b
        self.c = c
        self.init = init
        self.eval_func = 'RMSE'
    
    def f_x(self,x,y,z):
        return self.a*(y-x)
    def f_y(self,x,y,z):
        return (self.c-self.a)*x-x*z+self.c*y
    def f_z(self,x,y,z):
        return x*y-self.b*z
    
    def Runge_Kutta(self,x,y,z,step):
        k_x_1 = self.f_x(x,y,z)
        k_y_1 = self.f_y(x,y,z)
        k_z_1 = self.f_z(x,y,z)
        
        k_x_2 = self.f_x(x+1/(2*step)*k_x_1,y+1/(2*step)*k_y_1,z+1/(2*step)*k_z_1)
        k_y_2 = self.f_y(x+1/(2*step)*k_x_1,y+1/(2*step)*k_y_1,z+1/(2*step)*k_z_1)
        k_z_2 = self.f_z(x+1/(2*step)*k_x_1,y+1/(2*step)*k_y_1,z+1/(2*step)*k_z_1)
        
        k_x_3 = self.f_x(x+1/(2*step)*k_x_2,y+1/(2*step)*k_y_2,z+1/(2*step)*k_z_2)
        k_y_3 = self.f_y(x+1/(2*step)*k_x_2,y+1/(2*step)*k_y_2,z+1/(2*step)*k_z_2)
        k_z_3 = self.f_z(x+1/(2*step)*k_x_2,y+1/(2*step)*k_y_2,z+1/(2*step)*k_z_2)
        
        k_x_4 = self.f_x(x+1/step*k_x_3,y+1/step*k_y_3,z+1/step*k_z_3)
        k_y_4 = self.f_y(x+1/step*k_x_3,y+1/step*k_y_3,z+1/step*k_z_3)
        k_z_4 = self.f_z(x+1/step*k_x_3,y+1/step*k_y_3,z+1/step*k_z_3)
        
        x_next = x+1/(6*step)*(k_x_1+2*k_x_2+2*k_x_3+k_x_4)
        y_next = y+1/(6*step)*(k_y_1+2*k_y_2+2*k_y_3+k_y_4)
        z_next = z+1/(6*step)*(k_z_1+2*k_z_2+2*k_z_3+k_z_4)
        
        return x_next, y_next, z_next

    def generate_data(self, xyz, T_skip=100, T_train=8900, T_test=3000, step=100):
        T = T_skip+T_train+T_test
        
        x = np.array([self.init[0]])
        y = np.array([self.init[1]])
        z = np.array([self.init[2]])
        for i in range(T):
            x_next,y_next,z_next = self.Runge_Kutta(x[i],y[i],z[i],step)
            x = np.append(x,x_next)
            y = np.append(y,y_next)
            z = np.append(z,z_next)
            
        x_plus = x[1:]
        y_plus = y[1:]
        z_plus = z[1:]
        
        u = x[0:T]
        u = np.vstack((u, y[0:T]))
        u = np.vstack((u, z[0:T]))
            
        if(xyz=='x'):
            u_plus = x_plus
        elif(xyz=='y'):
            u_plus = y_plus
        elif(xyz=='z'):
            u_plus = z_plus
        else:
            raise ValueError
        
        return u.T, u_plus, [T_skip,T_train,T_test], self.eval_func

class Henon:
    def __init__(self, a=1.4, b=0.3, init=[0.1,0]):
        self.a = a
        self.b = b
        self.init = init
        self.eval_func = 'RMSE'
    
    def f_x(self,x,y):
        return 1-self.a*x**2+y
    def f_y(self,x,y):
        return self.b*x


    def generate_data(self, xyz, T_skip=100, T_train=8900, T_test=3000):
        T = T_skip+T_train+T_test
        
        x = np.array([self.init[0]])
        y = np.array([self.init[1]])
        for i in range(T):
            x_next,y_next = self.f_x(x[i],y[i]),self.f_y(x[i],y[i])
            x = np.append(x,x_next)
            y = np.append(y,y_next)
            
        x_plus = x[1:]
        y_plus = y[1:]
        
        u = x[0:T]
        u = np.vstack((u, y[0:T]))
            
        if(xyz=='x'):
            u_plus = x_plus
        elif(xyz=='y'):
            u_plus = y_plus
        else:
            raise ValueError
        
        return u.T, u_plus, [T_skip,T_train,T_test], self.eval_func

class COVID19:
    def __init__(self):
        self.eval_func = 'RMSE'
    
    def generate_data(self, country, xyz, T_skip=10, T_test=50):
        
        data = np.loadtxt(f'./dataset/{country}.csv',delimiter=',',skiprows=1)
        cases = data[:,0]
        deaths = data[:,1]
        
        T = len(cases)
        
        cases_plus = cases[7:]
        deaths_plus = deaths[7:]
        
        u = cases[0:T-7]
        u = np.vstack((u, deaths[0:T-7]))

        T_train = len(cases_plus) - T_skip - T_test
            
        if(xyz=='cases'):
            u_plus = cases_plus
        elif(xyz=='deaths'):
            u_plus = deaths_plus
        else:
            raise ValueError
        
        return u.T, u_plus, [T_skip,T_train,T_test], self.eval_func