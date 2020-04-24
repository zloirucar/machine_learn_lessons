from sklearn.svm import SVC
import pandas
from sklearn.metrics import accuracy_score
import numpy as np
data = pandas.read_csv('svm-data.csv', header=None)
y = data.iloc[:,0]
X = data.iloc[:,1:]
clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(X,y)
ans1 = open('ans1', 'w')
for i in clf.support_:
    ans1.write(str(i))