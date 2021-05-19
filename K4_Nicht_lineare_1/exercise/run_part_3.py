import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys

sys.path.insert(0, "..")
from exercise.utility import plot_cluster


def get_setosa_classifier_target_vector(y: np.ndarray):
    """Returns the target vector for a iris setosa classifier with shape (N,).
    I.e. the vector at position i is 1, if that data point is of class iris setosa, 0 otherwise.
    The class label for iris setosa is 0.
    y is a vector of class labels with shape (N,), where N is the number of data points."""
    y_setosa = np.zeros(y.shape)
    return y_setosa


def get_versicolor_classifier_target_vector(y: np.ndarray):
    """Returns the target vector for a iris versicolor classifier with shape (N,).
    I.e. the vector at position i is 1, if that data point is of class iris versicolor, 0 otherwise.
    The class label for iris versicolor is 1.
    y is a vector of class labels with shape (N,), where N is the number of data points."""
    y_versicolor = np.zeros(y.shape)
    return y_versicolor


def get_virginica_classifier_target_vector(y: np.ndarray):
    """Returns the target vector for a iris virginica classifier with shape (N,).
    I.e. the vector at position i is 1, if that data point is of class iris virginica, 0 otherwise.
    The class label for iris virginica is 2.
    y is a vector of class labels with shape (N,), where N is the number of data points."""
    y_virginica = np.zeros(y.shape)
    return y_virginica


def train_svm_clf_list(svm_clf_list: list, X: np.ndarray, y_class: list):
    """Train each classifier in svm_clf_list for the feature set X and the corresponding
    target column in y_class.
    Does not return anything."""
    pass


def optional_3():
    pass


if __name__ == "__main__":
    iris = load_iris()
    # now we look at all classes, but only at features 2, 3
    X = iris["data"][:, (2, 3)]
    # scale data. SVMs are sensitive to varying feature scales
    X = StandardScaler().fit_transform(X)
    y = iris["target"]
    # plot cluster
    plot_cluster(X, y, "petal length", "petal width", "Multi-Class Iris")

    # Since SVMs are binary classifiers we need a strategy for multi-class classification.
    # One such strategy is One-versus-Rest (OvR) which defines one classifier for each class.
    # To classify a new data point x, x is classified by every classifier (one for each class).
    # The result is a vector which determines for each class if x is of that class (i-th position is 1)

    # Since the iris data set has 3 classes, we need 3 classifiers.
    # Since each classifier needs to be trained for a different class, we need different target vectors for each class
    y_class = [get_setosa_classifier_target_vector(y),
               get_versicolor_classifier_target_vector(y),
               get_virginica_classifier_target_vector(y)]
    # One classifier for each class
    svm_clf_list = [SVC(kernel="linear", C=1, random_state=42),
                    SVC(kernel="poly", C=100, degree=2, random_state=42),
                    SVC(kernel="linear", C=1, random_state=42)]
    # Train each classifier
    train_svm_clf_list(svm_clf_list, X, y_class)

    # To get the result we need to make a prediction with each classifier
    prediciton = np.column_stack([clf.predict(X) for clf in svm_clf_list])
    print(prediciton)
