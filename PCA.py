import pandas
import numpy as np
from sklearn.decomposition import PCA

data = pandas.read_csv('close-prices.csv', index_col= 'date')
dj = pandas.read_csv('djia-index.csv', index_col = 'date')
pca = PCA(n_components=10)
pca.fit(data)

summa = 0
for i in range(0, len(pca.explained_variance_ratio_)):
    val = pca.explained_variance_ratio_[i]
    summa += val;
    if summa > 0.9:
        print(f"Need {i + 1} components!")
        break
f_c = pandas.DataFrame(pca.transform(data)[:,0])
coef = np.corrcoef(f_c.T, dj.T)[1,0]
print("Corrcoef:", round(coef, 2))
print("Company:" ,data.columns[np.argmax(pca.components_[0])])


print('Done!')