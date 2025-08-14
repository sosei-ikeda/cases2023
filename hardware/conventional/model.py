import numpy as np
from layer import Mask, DFR, Output, Tikhonov

class RC:
    def __init__(self, N_u, N_x, N_y, p, theta, gamma, eta, beta):
        self.N_u = N_u
        self.N_x = N_x
        self.Input = Mask(N_u,N_x,gamma)
        self.Reservoir = DFR(eta,p,N_x,theta)
        self.Output = Output(N_x+1, N_y)
        
        self.N_y = N_y
        self.y_prev = np.zeros(N_y)
        self.beta = beta
        
    def progress(self, i, total):
        if((i+1)%int(total/10)==0):
            # print(f'{i+1}/{total}')
            pass

    def train(self, U, D, skip):
        self.Reservoir.refresh()
        optimizer = Tikhonov(self.N_x+1, self.N_y, self.beta)  
        for i in range(len(U)):
            j = self.Input(U[i])
            x = self.Reservoir(j)
            if(i>=skip):
                d = D[i]
                optimizer(d, np.append(1,x))
            self.progress(i, len(U))
        self.Output.setweight(optimizer.get_Wout_opt())
        return x
    
    def predict(self, U):
        Y_pred = []
        for i in range(len(U)): 
            j = self.Input(U[i])
            x = self.Reservoir(j)
            y_pred = self.Output(np.append(1,x))
            Y_pred.append(y_pred)      
            self.progress(i, len(U))
        return np.array(Y_pred)