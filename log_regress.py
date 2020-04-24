import pandas
import numpy as np
import math
import sklearn.metrics
from sklearn.metrics import roc_auc_score
data = pandas.read_csv('data-logistic.csv', header=None)
y = data.iloc[:,0]
X = data.iloc[:,1:]

def dist(a, b):
    return np.sqrt(np.square(a[0]-b[0]) + np.square(a[1]-b[1]))

def log_regress(X, y, max_iter, w1 = 0, w2 = 0, C = 0, k = 0.1, dist_min = 0.00001):
    l = len(y)
    for j in range(max_iter):
        sum1 = 0
        sum2 = 0
        for i in range(l):
            s = 1 - 1/(1 + np.exp( -y[i] * (w1 * X[1][i] + w2 * X[2][i])))
            sum1 += y[i] * X[1][i] * s
            sum2 += y[i] * X[2][i] * s
        w1_n = w1 + k/l * sum1 - k * C * w1
        w2_n = w2 + k/l * sum2 - k * C * w2
        print('Шаг', j, 'евклид. расстояние: %0.6f' %dist([w1, w2], [w1_n, w2_n]))
        print('w1=%.8f, w2=%.8f' %(w1, w2)) 
        if dist([w1, w2], [w1_n, w2_n]) < dist_min:
            break
        w1, w2 = w1_n, w2_n
    return w1, w2

def auc_roc(X, y, w1=0, w2=0):
    l = len(y)
    a = []
    for i in range(l):
        a.append( 1 / (1 + np.exp( - w1*X[1][i] - w2*X[2][i])) )
    return a

ww1, ww2 = log_regress(X, y, max_iter = 1000, C = 0)
print('w1=%.8f, w2=%.8f' %(ww1, ww2))
print(roc_auc_score(y, auc_roc(X, y, ww1, ww2)))

ww1, ww2 = log_regress(X, y, max_iter = 100, C = 10)
print('w1=%.8f, w2=%.8f' %(ww1, ww2))
print(roc_auc_score(y, auc_roc(X, y, ww1, ww2)))

