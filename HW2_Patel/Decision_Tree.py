import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

train_cols = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "Output"]
train_file = pd.read_csv("train.csv", header=None)
train_file.columns = train_cols
test_file = pd.read_csv("test.csv", header=None)
test_cols = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]
test_file.columns = test_cols

X = train_file[test_cols]
y = train_file.Output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Model Accuracy, how often is the classifier correct?
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.f1_score(y_test, y_pred))

# -------------- {BEGIN} Decision Tree Classification --------------------
dt = DecisionTreeClassifier(criterion='entropy')
dt = dt.fit(X,y)
y_pred_tree = dt.predict(test_file)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt,
            feature_names=X.columns,
            filled=True)
fig.savefig("decision-tree.png")

# Decision Tree with Entropy is performing poorly as compared to Gini Index
# -------------- {END} Decision Tree Classification ----------------------


# -------------- {BEGIN} Random Forest Classification --------------------
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(X,y)
y_pred_forest = forest.predict(test_file)

# ------------- {END} Random Forest Classification ------------------------


# Writes all the scores from Tree(s) algorithm to a format.txt file.
with open('format.txt', 'w', encoding="utf-8") as finalFile:
    finalFile.writelines("%s\n" % place for place in y_pred_forest)