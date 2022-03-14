# IMPORTS: GENERAL
import numpy as np
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORTS: CLASSIFIER
from sklearn import neighbors

# IMPORTS: NOT USED
import pickle
from sklearn import preprocessing

# READING DATA
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

## IMPORTANT VARIABLES
CLASSIFIER_NAME = "KNN"
K = 5
CLASSIFIER_PARAMETERS_NAMES = ['K']
CLASSIFIER_PARAMETERS = [K]
NREPEATS = 1000
TEST_SIZE = 0.2

def knn(kind):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=TEST_SIZE, stratify=y)
    clf = neighbors.KNeighborsClassifier(n_neighbors=K) 
    clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)
    diff = y_prediction - y_test
    diff = list(diff)
    exact_accuracy = diff.count(0) / len(diff)
    off_accuracy = exact_accuracy + (diff.count(1) / len(diff)) + (diff.count(-1) / len(diff))

    if kind == 'exact':
        return exact_accuracy
    elif kind == 'off':
        return off_accuracy








# MAIN EXPERIMENT
input(f"""
number of features: {len(X.columns)}
featureset: {list(X.columns)}
classifier: {CLASSIFIER_NAME}
parameters: {CLASSIFIER_PARAMETERS_NAMES}
parameters: {CLASSIFIER_PARAMETERS}
repeats: {NREPEATS}
test_size: {TEST_SIZE}

Press Enter to confirm or ctrl + c to stop the program: """)
results = []
results_off = []
for i in range(NREPEATS):
    print(i)
    results.append(knn('exact'))
    results_off.append(knn('off'))
results = [result*100 for result in results]
results_off = [result*100 for result in results_off]

def print_results():
    print(f'--exact--')
    print(f'mean: {round(sum(results)/len(results), 1)}')
    print(f'sd: {round(np.std(results), 1)}')
    print(f'min: {round(min(results), 1)}')
    print(f'max: {round(max(results), 1)}')
    print()
    print(f'--off by one level--')
    print(f'mean: {round(sum(results_off)/len(results_off), 1)}')
    print(f'sd: {round(np.std(results_off), 1)}')
    print(f'min: {round(min(results_off), 1)}')
    print(f'max: {round(max(results_off), 1)}')

bins5 = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
def show_hist():
    plt.hist(results, edgecolor='black', bins=bins5)
    plt.title(f"Histogram plot of accuracy of the {CLASSIFIER_NAME} classifier across {NREPEATS} training sessions, k = {K}")
    plt.xlabel('Accuracy %')
    plt.show()

    plt.hist(results_off, edgecolor='black', bins=bins5)
    plt.title(f"Histogram plot of Off-by-one-level accuracy of the {CLASSIFIER_NAME} classifier across {NREPEATS} training sessions, k = {K}")
    plt.xlabel('Accuracy %')
    plt.show()

# Save data to .csv file
all_results = pd.DataFrame(data=[results, results_off]).T
all_results.columns=['exact', 'off']
all_results.to_csv(f"{CLASSIFIER_NAME} {CLASSIFIER_PARAMETERS_NAMES[0]}={CLASSIFIER_PARAMETERS[0]} N={NREPEATS} split={TEST_SIZE}.csv")

# FINAL SHOW
print_results()
show_hist()


