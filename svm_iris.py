
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

def plot_decision_boundary(model, x, y):

    plt.figure()

    # plot decision regions
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z, alpha=0.4)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolor='k')
    # plt.legend()

# import some data to play with
iris = load_iris()

df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])

# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica

# x = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
x = df[['sepal length (cm)', 'sepal width (cm)']].values
# scaler = StandardScaler()
# scaler.fit(x)
# x_scaled = scaler.transform(x)

y = df['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=4)

linear_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
))
linear_svm_clf.fit(x_train, y_train)

y_pred = linear_svm_clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print('linear svm accuracy: ' + str(acc))

plot_decision_boundary(linear_svm_clf, x_train, y_train)

polynomial_svm_clf = Pipeline((
    ("poly_features", PolynomialFeatures(degree=2)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ))
polynomial_svm_clf.fit(x_train, y_train)

y_pred = polynomial_svm_clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print('polynomial svm accuracy: ' + str(acc))
plot_decision_boundary(polynomial_svm_clf, x_train, y_train)

poly_kernel_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
))
poly_kernel_svm_clf.fit(x_train, y_train)

y_pred = poly_kernel_svm_clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print('polynomial svm accuracy: ' + str(acc))
plot_decision_boundary(poly_kernel_svm_clf, x_train, y_train)

rbf_kernel_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=1))
))
rbf_kernel_svm_clf.fit(x_train, y_train)
plot_decision_boundary(rbf_kernel_svm_clf, x_train, y_train)

y_pred = rbf_kernel_svm_clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print('rbf svm accuracy: ' + str(acc))

plt.show()