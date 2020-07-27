
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree

# import some data to play with
iris = load_iris()

df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])

# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica

x = df[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']].values
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

y = df['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=3)

model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)
prediction = model.predict(x_test)

error_rate = np.mean(prediction!=y_test)

#metrics.plot_confusion_matrix(NB_classifier, y_test.reshape(-1, 1), prediction)

print(error_rate)

plt.show()