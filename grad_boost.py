import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import math

data = pd.read_csv('gbm-data.csv')
y = data.iloc[:,0]
X = data.iloc[:,1:len(data.columns)]
X = np.array(X.values)
y = np.array(y.values)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.8, random_state=241)
clf = GradientBoostingClassifier(\
    n_estimators=50, verbose = True, learning_rate = 0.2, random_state = 241)
clf.fit(X_train, y_train)

def sigmoid(y_pred):
    return 1 / (1 + np.exp(-y_pred))

train_loss = np.empty(len(clf.estimators_))
test_loss = np.empty(len(clf.estimators_))

min_count = 0
min_arg = -1
for i, y_pred in enumerate(clf.staged_decision_function(X_train)):
    train_loss[i] = log_loss(y_train, sigmoid(y_pred))


for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    test_loss[i] = log_loss(y_test, sigmoid(y_pred))
    if min_arg > test_loss[i] or min_arg < 0:
        min_arg = test_loss[i]
        min_count = i
        print (f"For test min_arg = {round(min_arg, 2)}, min_count = {i}")

plt.figure()
plt.plot(test_loss, 'r', linewidth=2)
plt.plot(train_loss, 'g', linewidth=2)
plt.legend(['test', 'train'])
plt.show()

clf = GradientBoostingClassifier(\
    n_estimators=36, verbose = True, learning_rate = 0.2, random_state = 241)
clf.fit(X_train, y_train)
print(log_loss(y_test, clf.predict_proba(X_test)))

print('Done!')