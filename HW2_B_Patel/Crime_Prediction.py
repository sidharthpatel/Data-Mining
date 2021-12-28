import pandas as pd
import numpy as np
from numpy import mean, std
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pandas import Series
from sklearn import metrics, linear_model, datasets
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

train_file = pd.read_csv("train.csv")
test_file = pd.read_csv("test.csv")
# print(train_file.head())
def preprocess(file):
       # onehot = OneHotEncoder(sparse=False)
       vec = LabelEncoder()
       file['sex'] = vec.fit_transform(file['sex'])
       # temp = np.array(file['sex'])
       # temp = temp.reshape(len(temp), 1)
       # encoding = onehot.fit_transform(temp)
       # file['sex'] = encoding

       file['age_cat'] = vec.fit_transform(file['age_cat'])
       # temp = np.array(file['age_cat'])
       # temp = temp.reshape(len(temp), 1)
       # encoding = onehot.fit_transform(temp)
       # file['age_cat'] = encoding

       file['race'] = vec.fit_transform(file['race'])
       # temp = np.array(file['race'])
       # temp = temp.reshape(len(temp), 1)
       # encoding = onehot.fit_transform(temp)
       # file['race'] = encoding

       file['c_charge_degree'] = vec.fit_transform(file['c_charge_degree'])
       # temp = np.array(file['c_charge_degree'])
       # temp = temp.reshape(len(temp), 1)
       # encoding = onehot.fit_transform(temp)
       # file['c_charge_degree'] = encoding

       file['c_charge_desc'] = vec.fit_transform(file['c_charge_desc'])
       # temp = np.array(file['c_charge_desc'])
       # temp = temp.reshape(len(temp), 1)
       # encoding = onehot.fit_transform(temp)
       # file['c_charge_desc'] = encoding

preprocess(train_file)
preprocess(test_file)

# print(train_file.head())

X = train_file[['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count',
       'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc']]
y = train_file['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

s_classifier = SVC(kernel='rbf', random_state=1)
s_classifier.fit(X, y)
y_pred_svc = s_classifier.predict(test_file)

svc = SVC(kernel='rbf', random_state=1)
svc.fit(X_train, y_train)
y_pred_dummy = svc.predict(X_test)
cm = metrics.confusion_matrix(y_test,y_pred_dummy)
accuracy = float(cm.diagonal().sum())/len(y_test)
# print("\nAccuracy Of SVM: ", accuracy)

# Writes all the scores from Tree(s) algorithm to a format.txt file.
# with open('format.txt', 'w', encoding="utf-8") as finalFile:
#     finalFile.writelines("%s\n" % place for place in y_pred_svc)

# ----------- HOMEWORK 3 START --------------
calibrated = CalibratedClassifierCV(svc, method='sigmoid', cv=5)

# predicted = cross_val_predict(svc, X, y, cv=5)
predicted = cross_val_predict(calibrated, X, y, cv=5)
merge = np.array([y, train_file.race, predicted])
merge = np.transpose(merge)
data_frame = pd.DataFrame(merge, columns=['label', 'race', 'prediction'])
results = confusion_matrix(y_true=y, y_pred=predicted)

af_fp, ca_fp, hs_fp, other_fp, na_fp, a_fp = 0, 0, 0, 0, 0, 0

af_tp, ca_tp, hs_tp, other_tp, na_tp, a_tp = 0, 0, 0, 0, 0, 0

af_tn, ca_tn, hs_tn, other_tn, na_tn, a_tn = 0, 0, 0, 0, 0, 0

af_fn, ca_fn, hs_fn, other_fn, na_fn, a_fn = 0, 0, 0, 0, 0, 0

# false positive
for index in data_frame.index:
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 0):
              af_fp = af_fp + 1
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 2):
              ca_fp = ca_fp + 1
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 3):
              hs_fp = hs_fp + 1
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 5):
              other_fp = other_fp + 1
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 4):
              na_fp = na_fp + 1
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 1):
              a_fp = a_fp + 1
# true positive
for index in data_frame.index:
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 0):
              af_tp = af_tp + 1
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 2):
              ca_tp = ca_tp + 1
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 3):
              hs_tp = hs_tp + 1
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 5):
              other_tp = other_tp + 1
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 4):
              na_tp = na_tp + 1
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 1 and data_frame['race'][index] == 1):
              a_tp = a_tp + 1
# true negative
for index in data_frame.index:
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 0):
              af_tn = af_tn + 1
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 2):
              ca_tn = ca_tn + 1
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 3):
              hs_tn = hs_tn + 1
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 5):
              other_tn = other_tn + 1
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 4):
              na_tn = na_tn + 1
       if (data_frame['label'][index] == 0 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 1):
              a_tn = a_tn + 1
#false negative
for index in data_frame.index:
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 0):
              af_fn = af_fn + 1
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 2):
              ca_fn = ca_fn + 1
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 3):
              hs_fn = hs_fn + 1
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 5):
              other_fn = other_fn + 1
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 4):
              na_fn = na_fn + 1
       if (data_frame['label'][index] == 1 and data_frame['prediction'][index] == 0 and data_frame['race'][index] == 1):
              a_fn = a_fn + 1
print("African-American False Positives: ", af_fp)
print("Caucasian False Positives: ", ca_fp)
print("Hispaninc", hs_fp)
print("other", other_fp)
print("Native", na_fp)
print("Asian", a_fp)
print()
print("African-American True Positives: ", af_tp)
print("Caucasian True Positives: ", ca_tp)
print("Hispaninc", hs_tp)
print("other", other_tp)
print("Native", na_tp)
print("Asian", a_tp)
print()
print("African-American True Negative: ", af_tn)
print("Caucasian True Negative: ", ca_tn)
print("Hispaninc", hs_tn)
print("other", other_tn)
print("Native", na_tn)
print("Asian", a_tn)
print()
print("False Positive Numerator for Black Person: ", af_fp)
print("False Positive Denominator for Black Person: ", af_fp + af_tn)
print("False Positive Rate for Black Person: ", af_fp / (af_fp + af_tn))
print("False Positive Numerator for Caucasian Person: ", ca_fp)
print("False Positive Denominator for Caucasian Person: ", ca_fp + ca_tn)
print("False Positive Rate for Caucasian Person: ", ca_fp / (ca_fp + ca_tn))
print()
print("African-American False Negative: ", af_fn)
print("Caucasian False Negative: ", ca_fn)
print("Hispaninc", hs_fn)
print("other", other_fn)
print("Native", na_fn)
print("Asian", a_fn)
print()
print("True Positive Numerator for Black Person: ", af_tp)
print("True Positive Denominator for Black Person: ", af_tp + af_fn)
print("True Positive Rate for Black Person: ", af_tp / (af_tp + af_fn))
print("True Positive Numerator for Caucasian Person: ", ca_tp)
print("True Positive Denominator for Caucasian Person: ", ca_tp + ca_fn)
print("True Positive Rate for Caucasian Person: ", ca_tp / (ca_tp + ca_fn))

print(results)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()

X = train_file[['sex', 'age', 'age_cat', 'juv_fel_count', 'juv_misd_count',
       'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc']]

predicted = cross_val_predict(svc, X, y, cv=5)
cm = metrics.confusion_matrix(y, predicted)
accuracy = float(cm.diagonal().sum())/len(y)
print('Accuracy w/o Race Feature: ', accuracy)

# calibrated = CalibratedClassifierCV(svc, method='sigmoid', cv=3)
# predicted = cross_val_predict(calibrated, X, y, cv=5)
# cm = metrics.confusion_matrix(y, predicted)
# accuracy = float(cm.diagonal().sum())/len(y)
# print('Accuracy w/o Race Feature: ', accuracy)

# ----------- HOMEWORK 3 END ----------------