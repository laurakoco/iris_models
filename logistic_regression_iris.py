
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris

scaler = StandardScaler()

def multiclass(df):

    x = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
    scaler.fit(x)
    x = scaler.transform(x)

    y = df['target'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=4)

    logistic_regression = LogisticRegression().fit(x_train, y_train)
    pred = logistic_regression.predict(x_test)
    pred_prob = logistic_regression.predict_proba(x_test)
    acc = metrics.accuracy_score(y_test, pred)
    print('accuracy: ' + str(round(acc, 4)))
    # print('prob: ' + str(pred_prob))

def one_vs_all(df):

    x = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
    scaler.fit(x)
    x = scaler.transform(x)

    y = df['target'].values

    # setosa vs all
    for i in range(0,len(y)):
        if y[i] == 2:
            y[i] = 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=4)

    logistic_regression = LogisticRegression().fit(x_train, y_train)
    pred = logistic_regression.predict(x_test)
    pred_prob = logistic_regression.predict_proba(x_test)
    acc = metrics.accuracy_score(y_test, pred)
    print('accuracy: ' + str(round(acc, 4)))
    print('prob: ' + str(pred_prob))

def plot(df):

    x = df['sepal length (cm)'].tolist()
    y = df['sepal width (cm)'].tolist()
    label = df['target']

    df['target_name'] = 'nan'

    color_list = []
    for i in label:
        if i == 0:
            color_list.append('red')
        elif i == 2:
            color_list.append('blue')
        else:
            color_list.append('yellow')

    df['Color'] = color_list

    color_list = ['red', 'yellow', 'blue']

    plt.figure()

    for color in color_list:
        x = df[df['Color'] == color]['sepal length (cm)'].tolist()
        y = df[df['Color'] == color]['sepal width (cm)'].tolist()
        color_l = df[df['Color'] == color]['Color'].tolist()
        plt.scatter(x, y, color=color_l, label=color)

    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')

    plt.grid()
    plt.legend(['Setosa', 'Versicolour', 'Virginica'])

# import some data to play with
iris = load_iris()

df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns= iris['feature_names'] + ['target'])

# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica

multiclass(df)

plot(df)

# one_vs_all(df)
# plot(df)

plt.show()
