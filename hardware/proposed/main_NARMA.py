import numpy as np
from model import RC
from benchmark import NARMA

if __name__ == "__main__":
    T_skip = 100
    T_train = 5900
    T_test = 4000

    order = 10
    dynamics = NARMA(order, a1=0.3, a2=0.05, a3=1.5, a4=0.1)
    y_init = [0] * order
    u, d = dynamics.generate_data(T_skip+T_train+T_test, y_init)
    
    reservoir_bit = 24
    
    output_bit = reservoir_bit + 5
    k = 2**(reservoir_bit-2)/np.max(u)
    j = 2**(output_bit-1)/np.max(d)
    
    N_u = 1
    N_x = 100
    N_y = 1
    A = 0
    B = 0
    P_out = 4
    Q_out = 5
    R_out = 11
    P_fb = 1
    Q_fb = 2
    R_fb = 4
    beta = 4e-12*k**2

    train_U = (u[:T_skip+T_train].reshape(-1, 1)*k)
    train_D = d[:T_skip+T_train].reshape(-1, 1)*j
    test_U = (u[T_skip+T_train:].reshape(-1, 1)*k)
    test_D = d[T_skip+T_train:].reshape(-1, 1)*j
    
    model = RC(N_u,N_x,N_y,A,B,P_out,Q_out,R_out,P_fb,Q_fb,R_fb,beta,reservoir_bit,output_bit)

    print('train start')
    train_X = model.train(train_U, train_D, T_skip)
    print('predict start (test)')
    test_Y = model.predict(test_U)
    test_NRMSE = np.linalg.norm(test_D-test_Y)/np.linalg.norm(test_D)
    print('test_NRMSE =', test_NRMSE)
    
    np.save('../HLS/proposed_24bit/U', test_U.astype(np.int32).reshape(-1))
    np.save('../HLS/proposed_24bit/Win', ((model.Input.Win+1)/2).astype(np.int16).reshape(-1))
    np.save('../HLS/proposed_24bit/X', train_X.astype(np.int32).reshape(-1))
    np.save('../HLS/proposed_24bit/Wout', model.Output.Wout.astype(np.int32).reshape(-1))
    np.save('../HLS/proposed_24bit/D', test_D.astype(np.int32).reshape(-1))