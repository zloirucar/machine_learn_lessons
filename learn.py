import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
i = 0
main_data = pd.read_csv('titanic.csv', index_col='PassengerId')
data = pd.read_csv('titanic.csv', index_col='PassengerId')
Age_data = data['Age']
Pclass_data = data['Pclass']
Fare_data = data['Fare']
Sex_data = data['Sex']

def drop_nan(ser, data):
    i = 1
    while(i < 891):
        if math.isnan(ser[i]):
            data = data.drop(i)
        i += 1
    return (data)
data = drop_nan(Age_data, data)
data = drop_nan(Fare_data, data)
data = drop_nan(Pclass_data, data)

data = data.loc[:, ['Age', 'Pclass', 'Fare', 'Sex', 'Survived']]
data_age = data.loc[:, 'Age']
data_pclass = data.loc[:, 'Pclass']
data_fare = data.loc[:, 'Fare']
data_sex = data.loc[:, 'Sex']
data_goal = data.loc[:, 'Survived']
data_len = int(data_age.shape[0])

arr_sex = []
arr_fare = []
arr_pclass = []
arr_age = []
arr_sign = []
arr_goal = []

while(i < data_len):
    if data_sex[data_sex.index[i]] == 'male':
        arr_sex.append(1)
    else:
        arr_sex.append(0)
    i += 1
i = 0
while(i < data_len):
   arr_sign.append([data_fare[data_fare.index[i]], data_pclass[data_pclass.index[i]], arr_sex[i], data_age[data_age.index[i]]])
   arr_goal.append(data_goal[data_goal.index[i]])
   i += 1
clf = DecisionTreeClassifier(random_state=241)
clf.fit(arr_sign, arr_goal)
importances = clf.feature_importances_
print(importances)