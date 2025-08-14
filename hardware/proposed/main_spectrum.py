import numpy as np
from sklearn.metrics import roc_auc_score
from model import RC
from benchmark import Spectrum

if __name__ == "__main__":
    T_skip = 20
    T_train = 49*T_skip
    T_test = 6102-(T_train+T_skip+T_skip)
    
    reservoir_bit = 16
    output_bit = reservoir_bit + 5
    
    N_u = 1
    N_x = 100
    N_y = 1
    A = 0
    B = 0
    P_out = 7
    Q_out = 9
    R_out = 11
    P_fb = 4
    Q_fb = 5
    R_fb = 6

    
    for noise in [10,15,20]:
        for ant in [2,4,6]:
            print('noise:',noise,', ant:',ant)
    
            dynamics = Spectrum(noise=noise, ant=ant)
            u, d = dynamics.generate_data()
            
            k = 2**(reservoir_bit-2)/np.max(u)
            j = 2**(output_bit-2)/np.max(d)
            beta = 1e+4*k**2
            
            train_U = u[:T_skip+T_train].reshape(-1, 1)*k
            train_D = d[:T_skip+T_train].reshape(-1, 1)*j
            test_U = u[T_skip+T_train:].reshape(-1, 1)*k
            test_D = d[T_skip+T_train:].reshape(-1, 1)*j
            
            model = RC(N_u,N_x,N_y,A,B,P_out,Q_out,R_out,P_fb,Q_fb,R_fb,beta,reservoir_bit,output_bit)
            train_X = model.train(train_U, train_D, T_skip)
            test_Y = model.predict(test_U)
    
            test_D = test_D.astype('bool').reshape(-1)
            print('AUC:%.3f'%roc_auc_score(test_D, test_Y))