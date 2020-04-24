import pandas
import numpy as np
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
passeng = data["Name"].value_counts().sum()
sex = data["Sex"].value_counts()
male = sex['male']
female = sex['female']
surv = data['Survived'].value_counts()
surv_1 = surv[1] / passeng * 100
surv_0 = surv[0] / passeng * 100
pclass = data['Pclass'].value_counts()
class_1 = pclass[1].sum() / passeng * 100
age = data['Age']
age_aver = age.sum() / passeng
names = data['Name']
i = 1
array_first = []
array_last = []
while i < int(passeng + 1):
    buf = names[i].split(", ")
    buf_last = buf[1].split(". ")
    if buf_last[0] == "Mrs" or buf_last[0] == "Miss":
        array_first.append(str(buf[0]))
        array_last.append(str(buf_last[0]))
        i += 1
    else:
        i += 1
d = {"first":array_last, "last":array_first}
df_dic = pandas.DataFrame(d)
ser_first = df_dic['last'].value_counts()

sib = data["SibSp"]
parch = data["Parch"]
sib_parch = {"sib":sib, "parch":parch}
df_sp = pandas.DataFrame(sib_parch)

print (male, female)
print("%.2f" % surv_1)
print("%.2f" % class_1)
print("%.2f" % age.mean(), "%.2f" % age.median())
print("%.2f" % df_sp.corr()["sib"][1])
print(str(ser_first.head(1)).split(' ')[0], end='')

