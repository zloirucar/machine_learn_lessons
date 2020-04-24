import pandas
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
data = np.genfromtxt('wine.data', delimiter=',')
data_name = data[:,0]
data_sign = data[:,1:]
result = {}
res_k = 0
ans1 = open("ans1","w")
ans2 = open("ans2","w")
ans3 = open("ans3","w")
ans4 = open("ans4","w")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for k in range(1, 51):
    classifier = KNeighborsClassifier(n_neighbors=k)
    quality = cross_val_score(classifier, data_sign, data_name, cv=kf, scoring='accuracy')
    result[k]=np.mean(quality)
for max_res in result.values():
    res_k += 1
    if max_res == max(result.values()):
        print('k =', res_k, 'result =', result[res_k])
        ans1.write(str(res_k))
        ans2.write(str(result[res_k]))
res_k = 0
for k in range(1, 51):
    classifier = KNeighborsClassifier(n_neighbors=k)
    quality = cross_val_score(classifier, preprocessing.scale(data_sign), data_name, cv=kf, scoring='accuracy')
    result[k]=np.mean(quality)
for max_res in result.values():
    res_k += 1
    if max_res == max(result.values()):
        print('k =', res_k, 'result =', result[res_k])
        ans3.write(str(res_k))
        ans4.write(str(result[res_k]))