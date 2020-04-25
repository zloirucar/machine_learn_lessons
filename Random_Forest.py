import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = data.iloc[:,0:len(data.columns) - 1]
y = data.iloc[:,len(data.columns) - 1]
for i in range(1, 51):
    classifier = RandomForestRegressor(random_state=1, n_estimators= i)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    quality = cross_val_score(classifier, X, y, cv=kf, scoring='r2')
    print(f"Quality: {round(np.mean(quality), 3)} \nCount: {i}")
    if round(np.mean(quality), 2) > 0.52:
        break
