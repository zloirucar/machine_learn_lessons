from sklearn.linear_model import Perceptron
import pandas
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
test = pandas.read_csv('perceptron-test.csv', header=None)
train = pandas.read_csv('perceptron-train.csv', header=None)
y_test = test.iloc[:,0]
X_test = test.iloc[:,1:]
y_train = train.iloc[:,0]
X_train = train.iloc[:,1:]
clf = Perceptron(random_state=241, max_iter=5, tol=None)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
before = accuracy_score(y_test, predictions)
print(before)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = Perceptron(random_state=241, max_iter=5, tol=None)
clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)
after = accuracy_score(y_test, predictions)
print(after)
print(after - before)