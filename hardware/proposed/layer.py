import numpy as np
from func import approx_in_signed_binary, right_shift
from func import signed_binary_to_decimal, decimal_to_signed_binary, exor

seed = 0

class Mask:
    def __init__(self, N_u, N_x, mask, alpha, seed=seed):
        np.random.seed(seed)
        if(mask == 'binary'):
            self.Win = (-1)**np.random.randint(0,2,(N_x,N_u))
        elif(mask=='triple'):
            self.Win = 0.5+0.09*(-1)**np.random.randint(0,2,(N_x,N_u))
        self.zero_arr = np.random.rand(N_x,N_u)<alpha
        self.Win = self.Win*self.zero_arr
        
    def __call__(self, u):
        return np.dot(self.Win, u)

class DFR:
    def __init__(self, N_x, A, B, P_out, Q_out, R_out, P_fb, Q_fb, R_fb, bit, reservoir):
        self.N_x = N_x
        self.A = A
        self.B = B
        self.P_out = P_out
        self.Q_out = Q_out
        self.R_out = R_out
        self.P_fb = P_fb
        self.Q_fb = Q_fb
        self.R_fb = R_fb
        self.x_prev = np.zeros(N_x)
        self.reservoir = reservoir
        self.bit = bit
    
    def f(self, x, j):
        a = right_shift(int(x),self.A,self.bit)
        b = right_shift(int(j),self.B,self.bit)
        if(self.reservoir == 'sum'):
            return np.frompyfunc(approx_in_signed_binary,2,1)(a+b,self.bit)
        elif(self.reservoir == 'exor'):
            a_bin = np.frompyfunc(decimal_to_signed_binary,2,1)(a,self.bit)
            b_bin = np.frompyfunc(decimal_to_signed_binary,2,1)(b,self.bit)
            exor_bin = np.frompyfunc(exor,2,1)(a_bin,b_bin)
            return np.frompyfunc(signed_binary_to_decimal,1,1)(exor_bin)
    
    def shift_sum(self, a, p, q, r):
        return right_shift(a,p,self.bit)+right_shift(a,q,self.bit)+right_shift(a,r,self.bit)
        
    def __call__(self, J):
        X = np.zeros(self.N_x)
        el_out = self.shift_sum(self.f(self.x_prev[0],J[0]),self.P_out,self.Q_out,self.R_out)
        el_fb = self.shift_sum(int(self.x_prev[self.N_x-1]),self.P_fb,self.Q_fb,self.R_fb)
        X[0] = np.frompyfunc(approx_in_signed_binary,2,1)(el_out+el_fb, self.bit)
        for i in range(self.N_x-1):
            el_out = self.shift_sum(self.f(self.x_prev[i+1],J[i+1]),self.P_out,self.Q_out,self.R_out)
            el_fb = self.shift_sum(int(X[i]),self.P_fb,self.Q_fb,self.R_fb)
            X[i+1] = np.frompyfunc(approx_in_signed_binary,2,1)(el_out+el_fb, self.bit)
        self.x_prev = X
        return self.x_prev
    
    def refresh(self):
        self.x_prev = np.zeros(self.N_x)

class Output:
    def __init__(self, N_r, N_y, res_bit, out_bit, seed=seed):
        np.random.seed(seed=seed)
        self.res_bit = res_bit
        self.out_bit = out_bit
        self.Wout = np.random.normal(size=(N_y, N_r))
    
    def __call__(self, r):
        return np.frompyfunc(approx_in_signed_binary,2,1)(np.dot(self.Wout, r), self.out_bit)
    
    def setweight(self, Wout_opt):
        self.Wout = np.frompyfunc(approx_in_signed_binary,2,1)(Wout_opt, self.out_bit)
        
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