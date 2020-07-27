
# naive bayesian classifier

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

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

NB_classifier = GaussianNB().fit(x_train,y_train)
prediction = NB_classifier.predict(x_test)
error_rate = np.mean(prediction!=y_test)

print(error_rate)