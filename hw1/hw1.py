import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
clf = tree.DecisionTreeClassifier(random_state=0)
order = pd.read_csv("train.csv")

pat_map = {'none': 0, 'some': 1, 'full': 2}
price_map = {'$': 1, '$$': 2, '$$$': 3}
type_map = {'french': 0, 'thai': 1, 'burger': 2, 'italian': 3}
est_map = {'0-10': 0, '10-30': 1, '30-60': 2, '>60': 3}

order["pat"] = order["pat"].map(pat_map)
order["price"] = order["price"].map(price_map)
order["type"] = order["type"].map(type_map)
order["est"] = order["est"].map(est_map)

X = order.drop(labels=['willWait',  'costunerId'], axis=1)
Y = order[['willWait']]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=1)
dtree =tree.DecisionTreeClassifier()
dtree.fit(Xtrain, Ytrain)
print("準確率 :", dtree.score(Xtest, Ytest))
preds= dtree.predict_proba(X=Xtest)

# plt.rcParams.update({"font.size": 10})

plt.figure(figsize=(10,5))
plot_tree(dtree, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()
