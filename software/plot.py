import numpy as np
import matplotlib.pyplot as plt

dataset = np.array(['Lorenz x','Lorenz y','Lorenz z','Rossler x','Rossler y','Rossler z','DST 2013','DST 2014'])
MOICBLS = np.array([2.591E-4,4.089E-4,4.766E-4,3.559E-4,1.210E-4,2.383E-3,3.456,7.663])
MGDFR = np.array([7.996E-5,3.219E-4,7.208E-4,2.256E-5,1.443E-5,1.391E-4,2.923,3.927])
identityDFR = np.array([1.353E-4,3.195E-3,3.986E-3,6.069E-6,3.851E-6,6.940E-4,2.898,3.964])
tanhDFR = np.array([2.916E-5,2.888E-4,6.281E-4,6.067E-6,2.841E-6,2.174E-4,2.818,3.914])

plt.figure(figsize=(7,4))
height = 0.2
x = np.arange(len(dataset))
line1 = plt.barh(x-height*1.2,-np.log10(tanhDFR/MOICBLS)[::-1],height,label='tanh DFR')
line2 = plt.barh(x,-np.log10(identityDFR/MOICBLS)[::-1],height,label='identity DFR')
line3 = plt.barh(x+height*1.2,-np.log10(MGDFR/MOICBLS)[::-1],height,label='MG DFR')
plt.yticks(x,dataset[::-1])
plt.xlabel('Ordinary Logarithm of Reciprocal of Ratio of RMSE to MOICBLS')
plt.ylabel('dataset')
plt.legend(handles=[line3,line2,line1], labels=['MG DFR','identity DFR','tanh DFR'],loc='lower right')
plt.savefig('RMSE_comparison.svg', bbox_inches='tight', pad_inches=0) 