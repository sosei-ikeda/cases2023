import numpy as np
from layer import Mask, DFR, Output, Tikhonov

class RC:
    def __init__(self, N_u, N_x, N_y, A, B, P_out, Q_out, R_out, P_fb, Q_fb, R_fb, beta, reservoir_bit, output_bit, mask='binary', alpha=1, reservoir='sum'):
        self.Input = Mask(N_u,N_x,mask,alpha)
        self.Reservoir = DFR(N_x,A,B,P_out,Q_out,R_out,P_fb,Q_fb,R_fb,reservoir_bit,reservoir)
        self.Output = Output(N_x+1,N_y,reservoir_bit,output_bit)
        
        self.N_u = N_u
        self.N_x = N_x
        self.N_y = N_y
        self.beta = beta
        
    def progress(self, i, total):
        if((i+1)%int(total/10)==0):
            # print(f'{i+1}/{total}')
            pass
    
    def train(self, U, D, skip=100):
        optimizer = Tikhonov(self.N_x+1, self.N_y, self.beta)  
        if(self.N_y==1): # regression
            for i in range(len(U)):
                j = self.Input(U[i]).astype(np.int64)
                x = self.Reservoir(j)
                if(i>=skip):
                    d = D[i]
                    optimizer(d, np.append(1,x))
                self.progress(i, len(U))
        else: # classification
            for i in range(len(U)):
                t = 0    
                while(1):
                    j = self.Input(U[i][t][:U.shape[2]-1]).astype(np.int64)
                    x = self.Reservoir(j)
                    d = D[i]
                    optimizer(d, np.append(x,1))
                    if(U[i][t][U.shape[2]-1]!=0):
                        break
                    t += 1
                self.Reservoir.refresh()
                self.progress(i, len(U))
        self.Output.setweight(optimizer.get_Wout_opt())
        return x
    
    def predict(self, U):
        Y_pred = []
        if(self.N_y==1): # regression
            for i in range(len(U)): 
                j = self.Input(U[i]).astype(np.int64)
                x = self.Reservoir(j)
                y_pred = self.Output(np.append(1,x))
                Y_pred.append(y_pred)      
                self.progress(i, len(U))
            self.Reservoir.refresh()
            return np.array(Y_pred)
        else: # classification
            Length = []
            for i in range(len(U)):
                t = 0    
                while(1):
                    j = self.Input(U[i][t][:U.shape[2]-1]).astype(np.int64)
                    x = self.Reservoir(j)
                    y_pred = self.Output(np.append(x,1))
                    Y_pred.append(y_pred)
                    if(U[i][t][U.shape[2]-1]!=0):
                        break
                    t += 1
                Length.append(t+1)
                self.Reservoir.refresh()
                self.progress(i, len(U))
            Y_pred = np.array(Y_pred)
    
            Label = np.empty(0,int)
            start = 0            
            for i in range(len(Length)):
                tmp = Y_pred[start:start+Length[i],:]
                max_index = np.argmax(tmp, axis=1)
                histogram = np.bincount(max_index)
                Label = np.hstack((Label, np.argmax(histogram)))
                start = start + Length[i]
            return Label

    def label(self, Y):
        Label = np.empty(0,int)
        for i in range(len(Y)):
            Label = np.hstack((Label, np.argmax(Y[i]))) 
        return Label
    
    def ACC(self, Label1, Label2):
        count = 0
        for i in range(len(Label1)):
            if Label1[i] == Label2[i]:
                count += 1
        return(count/len(Label1))