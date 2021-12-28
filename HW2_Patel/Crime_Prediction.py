import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

train_file = pd.read_csv("train.csv")
test_file = pd.read_csv("test.csv")
print(train_file.head())
def preprocess(file):
       onehot = OneHotEncoder(sparse=False)
       vec = LabelEncoder()
       file['sex'] = vec.fit_transform(file['sex'])
       temp = np.array(file['sex'])
       temp = temp.reshape(len(temp), 1)
       encoding = onehot.fit_transform(temp)
       file['sex'] = encoding

       file['age_cat'] = vec.fit_transform(file['age_cat'])
       temp = np.array(file['age_cat'])
       temp = temp.reshape(len(temp), 1)
       encoding = onehot.fit_transform(temp)
       file['age_cat'] = encoding

       file['race'] = vec.fit_transform(file['race'])
       temp = np.array(file['race'])
       temp = temp.reshape(len(temp), 1)
       encoding = onehot.fit_transform(temp)
       file['race'] = encoding

       file['c_charge_degree'] = vec.fit_transform(file['c_charge_degree'])
       temp = np.array(file['c_charge_degree'])
       temp = temp.reshape(len(temp), 1)
       encoding = onehot.fit_transform(temp)
       file['c_charge_degree'] = encoding

       file['c_charge_desc'] = vec.fit_transform(file['c_charge_desc'])
       temp = np.array(file['c_charge_desc'])
       temp = temp.reshape(len(temp), 1)
       encoding = onehot.fit_transform(temp)
       file['c_charge_desc'] = encoding

preprocess(train_file)
preprocess(test_file)

print(train_file.head())

X = train_file[['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count',
       'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc']]
y = train_file['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.f1_score(y_test, y_pred))

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(X,y)
y_pred_forest = forest.predict(test_file)

s_classifier = SVC(kernel='rbf', random_state=1)
s_classifier.fit(X, y)
y_pred_svc = s_classifier.predict(test_file)

svc = SVC(kernel='rbf', random_state=1)
svc.fit(X_train, y_train)
y_pred_dummy = svc.predict(X_test)
cm = metrics.confusion_matrix(y_test,y_pred)
accuracy = float(cm.diagonal().sum())/len(y_test)
print("\nAccuracy Of SVM: ", accuracy)

# Writes all the scores from Tree(s) algorithm to a format.txt file.
with open('format.txt', 'w', encoding="utf-8") as finalFile:
    finalFile.writelines("%s\n" % place for place in y_pred_svc)