import pandas
import numpy as np
import math
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
data = pandas.read_csv('classification.csv')
true = data.iloc[:,0]
predict = data.iloc[:,1]
TP = 0
FN = 0
FP = 0
TN = 0
for i in range(0, len(true)):
    if predict[i] == true[i] and true[i] == 1:
        TP += 1
    elif predict[i] > true [i]:
        FP += 1
    elif predict[i] < true[i]:
        FN += 1
    elif predict[i] == true[i]:
        TN += 1
ans1 = open("ans1","w")
ans1.write(str(TP) + ' ' + str(FP) + ' ' + str(FN) + ' ' + str(TN))

ans2 = open('ans2', 'w')
AS = accuracy_score(true, predict)
PS = precision_score(true, predict)
RS = recall_score(true, predict)
F1 = f1_score(true, predict)
ans2.write(str(round(AS, 2)) + ' ' + str(round(PS, 2)) + ' ' + str(round(RS, 2)) + ' ' + str(round(F1, 2)))

data = pandas.read_csv('scores.csv')
true = data.iloc[:,0]
logreg = data.iloc[:,1]
svm = data.iloc[:,2]
knn = data.iloc[:,3]
tree = data.iloc[:,4]
names = list(data.columns)
for i in range (0, 5):
    print(names[i])
    print(roc_auc_score(true, data.iloc[:,i]))
for i in range (0, 5):
    max_pres = 0
    max_recall = 0
    cur_data = data.iloc[:,i]
    precision, recall, thresholds = precision_recall_curve(true, cur_data)
    for j in range (0, len(precision)):
        if recall[j] > 0.7 and max_pres < precision[j]:
            max_pres = precision[j]
            max_recall = recall[j]
    print('Name:', names[i], 'Presicion:', round(max_pres, 3), 'Recall:', max_recall)

    
    

