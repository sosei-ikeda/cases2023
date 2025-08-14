import numpy as np

seed = 0

class NARMA:
    def __init__(self, m, a1, a2, a3, a4):
        self.m = m
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def generate_data(self, T, y_init, seed=seed):
        n = self.m
        y = y_init
        np.random.seed(seed=seed)
        u = np.random.uniform(0, 0.5, T)
        while n < T:
            y_n = self.a1*y[n-1] + self.a2*y[n-1]*(np.sum(y[n-self.m:n-1])) \
                + self.a3*u[n-self.m]*u[n] + self.a4
            y.append(y_n)
            n += 1
        return u, np.array(y)

class Spectrum:
    def __init__(self, noise, ant):
        self.noise = noise
        self.ant = ant

    def generate_data(self):
        spectrum_vector = np.genfromtxt(f"../dataset/spectrum/spectrum_-{self.noise}_db_{self.ant}_ant.csv", delimiter=",")
        u = spectrum_vector[:,0]
        d = spectrum_vector[:,1]
        return u, d