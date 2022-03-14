# IMPORTS: GENERAL
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# IMPORTS: CLASSIFIERS
from sklearn import svm

df = pd.read_csv('training_data_new.csv')
df.drop(['filename'], 1, inplace=True)

# FEATURESETS
featureset1 = df.drop(['cefr'], axis=1)
featureset2 = df.filter(['len', 'dcrs', 'ajcv', 'fkg', 'fre', 'ttr', 'bpera', 'ari', 'cli', 'asl', 'awl', 'asl.avps', 'avps'])
featureset3 = df.filter(['len', 'awl', 'asl'], axis=1)
featureset4 = df.drop(['cefr', 'len'], axis=1)
featureset5 = df.filter(['dcrs', 'ajcv', 'fkg', 'fre', 'ttr', 'bpera', 'ari', 'cli', 'asl', 'awl', 'asl.avps', 'avps'])
featureset6 = df.filter(['awl', 'asl'], axis=1)
X = featureset6 ############################# SELECTED FEATURESET ################################
y = np.array(df['cefr'])

y_df = pd.DataFrame(y)
entire_df = pd.DataFrame()
repeat_n = 3
rs_list = [74, 54, 80]
for rep in range(repeat_n):
        rs = rs_list[rep]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        split_xy = skf.split(X, y)
        
        counter = 1 # CAN REMOVE
        for i in split_xy:
                train, test = i # indexes for training and testing data
                X_train = X.iloc[train]
                y_train = y_df.iloc[train]
                X_test = X.iloc[test]
                y_test = y_df.iloc[test]

                # CLASSIFIER
                clf = svm.SVC(kernel='poly', degree=4, C=1000)
                clf.fit(X_train, y_train.to_numpy().flatten())
                #model.fit(X_train, y_train.to_numpy().flatten())
                y_prediction = clf.predict(X_test)
                y_prediction = np.round(y_prediction)

                # OFF ACCURACY
                diff = (y_prediction - np.array(y_test).flatten())
                diff_list = diff.tolist()
                diff_list = [abs(i) for i in diff_list]
                off_accuracy = (diff_list.count(0) + diff_list.count(1)) / len(diff_list)
                
                cr = classification_report(y_test, y_prediction, output_dict=True)
                cr_df = pd.DataFrame(cr)
                cr_df['off'] = off_accuracy
                entire_df = pd.concat([entire_df, cr_df])
                entire_df.loc[entire_df.shape[0]] = ['','','','','','','','','']
                entire_df.loc[entire_df.shape[0]] = ['','','','','','','','','']
                #cr_df.to_csv(f'lr_f1_round{rep+1}_rs{rs}_K{counter}.csv')
                counter += 1 # CAN REMOVE

entire_df.to_csv('SVMP_C=1000_D=4_FS6_rs74_rs54_rs80.csv')
