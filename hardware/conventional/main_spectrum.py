from sklearn.metrics import roc_auc_score
from model import RC
from benchmark import Spectrum

if __name__ == "__main__":
    T_skip = 20
    T_train = 49*T_skip
    T_test = 6102-(T_train+T_skip+T_skip)
    
    N_u = 1
    N_x = 100
    N_y = 1
    p = 1
    theta = 0.2
    gamma = 0.01
    eta = 0.01
    beta = 1e-3
    
    
    for noise in [10,15,20]:
        for ant in [2,4,6]:
            print('noise:',noise,', ant:',ant)
    
            dynamics = Spectrum(noise=noise, ant=ant)
            u, d = dynamics.generate_data()
            
            train_U = u[:T_skip+T_train].reshape(-1, 1)
            train_D = d[:T_skip+T_train].reshape(-1, 1)
            test_U = u[T_skip+T_train:].reshape(-1, 1)
            test_D = d[T_skip+T_train:].reshape(-1, 1)
            
            model = RC(N_u,N_x,N_y,p,theta,gamma,eta,beta)
            train_X = model.train(train_U, train_D, T_skip)
            test_Y = model.predict(test_U)
    
            test_D = test_D.astype('bool').reshape(-1)
            print('AUC:%.3f'%roc_auc_score(test_D, test_Y))