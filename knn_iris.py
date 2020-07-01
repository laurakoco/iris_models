
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris

# import some data to play with
iris = load_iris()

df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns= iris['feature_names'] + ['target'])

# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica

x = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

y = df['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=4)

k_array = np.arange(1,15,2)
acc_list = []
for k in k_array:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    acc = metrics.accuracy_score(y_test,y_pred)
    acc_list.append(acc)

plt.figure()
plt.plot(k_array,acc_list)
plt.grid()
plt.xlabel('k')
plt.ylabel('accuracy')