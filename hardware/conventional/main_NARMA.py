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
    
    train_U = u[:T_skip+T_train].reshape(-1, 1)
    train_D = d[:T_skip+T_train].reshape(-1, 1)
    test_U = u[T_skip+T_train:].reshape(-1, 1)
    test_D = d[T_skip+T_train:].reshape(-1, 1)
    
    N_u = 1
    N_x = 100
    N_y = 1
    p = 1
    theta = 0.2
    gamma = 0.05
    eta = 0.5
    beta = 1e-16

    model = RC(N_u,N_x,N_y,p,theta,gamma,eta,beta)
    
    print('train start')
    train_X = model.train(train_U, train_D, T_skip)
    print('predict start (test)')
    test_Y = model.predict(test_U)
    j = 2**(30)/np.max(test_Y)
    test_NRMSE = np.linalg.norm(test_D-test_Y)/np.linalg.norm(test_D)
    print('test_NRMSE =', test_NRMSE)
    
    k = 2**(30)/np.max(test_U)
    print(j,k)
    np.save('../HLS/conventional/U', (test_U*k).astype(np.int64).reshape(-1))
    np.save('../HLS/conventional/Win', model.Input.Win.reshape(-1))
    np.save('../HLS/conventional/X', train_X.reshape(-1))
    np.save('../HLS/conventional/Wout', model.Output.Wout.reshape(-1))
    np.save('../HLS/conventional/D', test_D.reshape(-1))