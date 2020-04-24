from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )
X = newsgroups.data
y = newsgroups.target
vectorizer = TfidfVectorizer()
Z = vectorizer.fit_transform(X)
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(Z, newsgroups.target)
print(gs.best_params_)
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(C=1.0, kernel='linear', random_state=241)
clf = clf.fit(Z, newsgroups.target)
result = np.argsort(np.abs(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
ans1 = open("ans1","w")
result_word = []
print(result)
for word in result:
    print(vectorizer.get_feature_names()[word])
    result_word.append(str(vectorizer.get_feature_names()[word]))
result_word.sort()
ans1.write(', '.join(result_word))
