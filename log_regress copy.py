import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

data = pd.read_csv('data-logistic.csv', header=None)
X = data.iloc[:,1:]
y = data.iloc[:,0]

w1 = 0   # вес 1
w2 = 0   # вес 2
k  = 0.1 # шаг
max_iter = 10000   # максимальное кол-во итераций
evk_min  = 0.00001 # минимальное евклидовое расcтояние

def logistic_reqression(X, y, w1=0, w2=0, c=0, k=0.1, max_iter=10000, evk_min=0.00001):
    l = len(y)
    for j in range(max_iter):
        summa1 = 0
        summa2 = 0
        for i in range(l):
            s = 1 - 1 / (1 + np.exp( -y[i] * (w1 * X[1][i] + w2 * X[2][i]) ))
            summa1 += y[i] * X[1][i] * s
            summa2 += y[i] * X[2][i] * s
        w1new = w1 + k / l * summa1 - k * c * w1
        w2new = w2 + k / l * summa2 - k * c * w2
        evk = np.sqrt( (w1new - w1)**2 + (w2new - w2)**2 )
        print('Шаг', j, 'евклид. расстояние: %0.6f' %evk)
        print('w1=%.8f, w2=%.8f' %(w1, w2))
        if ( evk < evk_min ): break
        w1, w2 = w1new, w2new
    return w1, w2

def auc_roc(X, y, w1=0, w2=0):
    l = len(y)
    a = []
    for i in range(l):
        a.append( 1 / (1 + np.exp( - w1*X[1][i] - w2*X[2][i])) )
    return roc_auc_score(y, a)

ww1, ww2 = logistic_reqression(X, y, c=0)
print('w1=%.8f, w2=%.8f' %(ww1, ww2))
#print ('C=0:', auc_roc(X, y, ww1, ww2))
print(auc_roc(X, y, ww1, ww2))

ww1, ww2 = logistic_reqression(X, y, c=10)
print('w1=%.8f, w2=%.8f' %(ww1, ww2))
#print ('C=10:', auc_roc(X, y, ww1, ww2))
print(auc_roc(X, y, ww1, ww2))
