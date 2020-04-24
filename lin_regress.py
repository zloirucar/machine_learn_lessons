import pandas
import numpy as np
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import coo_matrix, hstack


data = pandas.read_csv('salary-train.csv')
data_test = pandas.read_csv('salary-test.csv')
y = data.iloc[:,3:4]
X_train = data.iloc[:,0:3]
X_test = data_test.iloc[:,0:3]

X_train['FullDescription'] = X_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
X_train['LocationNormalized'] = X_train['LocationNormalized'].fillna('nan')
X_train['ContractTime'] = X_train['ContractTime'].fillna('nan')
X_train['FullDescription'] = X_train['FullDescription'].str.lower()

vectorizer = TfidfVectorizer(min_df = 5)
X_train_vec = vectorizer.fit_transform(X_train['FullDescription'])
X_test_vec = vectorizer.transform(X_test['FullDescription'])

enc = DictVectorizer()
X_train_categ = enc.fit_transform(X_train[['LocationNormalized', 'ContractTime']] \
    .to_dict('records'))
X_test_categ = enc.transform(X_test[['LocationNormalized', 'ContractTime']] \
    .to_dict('records'))

X_for_train = hstack([X_train_vec, X_train_categ])
X_for_test = hstack([X_test_vec, X_test_categ])

clf = Ridge(alpha=1.0, random_state=241)
clf.fit(X_for_train, y)
answer = clf.predict(X_for_test).round(decimals=2)
ans1 = open("ans1","w")
ans1.write(str(answer[0]) + ' ' + str(answer[1]))
print('DONE!')