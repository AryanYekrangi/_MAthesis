import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot

df = pd.read_csv('training_data_new.csv', header=0)
df.drop(['filename'], 1, inplace=True)
X = df.drop(['cefr'], axis=1)
y = np.array(df['cefr'])

# feature selection function
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

# scores for the features
labels = X.columns
for i in range(len(fs.scores_)):
    print(labels[i],fs.scores_[i])
# plot the scores
pyplot.bar(labels, fs.scores_)
pyplot.show()

