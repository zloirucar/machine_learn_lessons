import pandas
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.datasets import load_boston
data = load_boston(return_X_y=True)
X = data[0]
y = preprocessing.scale(data[1])
result = {}
for k in np.linspace(1, 10, num=200):
    classifier = KNeighborsRegressor(metric= 'minkowski', n_neighbors=5, weights='distance', p=k)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    quality = cross_val_score(classifier, X, y, cv=kf, scoring='neg_mean_squared_error')
    result[np.mean(quality)] = k
print(result[max(result)])