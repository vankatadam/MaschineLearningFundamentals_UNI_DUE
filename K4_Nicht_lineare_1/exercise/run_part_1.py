from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, "..")
from exercise.utility import plot_cluster, plot_svm


def optional_1():
    pass


def question_1():
    """Returns the value of C ∈ [0.1, 1, 100], for which the training score is best."""
    return -1


def question_2():
    """Returns the value of C ∈ [0.1, 1, 100], for which the test score is best."""
    return -1


if __name__ == "__main__":
    # load data set
    iris = load_iris()
    # to make it easier we only look at classes 1, 2 and features 2, 3
    X = iris["data"][iris["target"] != 0][:, (2, 3)]
    # SVMs are binary classifiers. Thus, they can only distinguish between class X and not class X.
    # Since our target data contains the class labels for each data point, we need to redefine it as:
    # 1 if the data point is of class x, 0 if the data point is not of class x
    # in this case the class x is iris virginica with the label 2
    y = (iris["target"][iris["target"] != 0] == 2).astype(float)
    # plot data points
    plot_cluster(X, y, "petal length", "petal width", "Versicolor (Blue) and Virginica (Brown)")
    # split data in to train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Create and train SVM
    svm_clf = SVC(C=0.1, kernel="linear", random_state=42)
    svm_clf.fit(X_train, y_train)
    # plot decision boundary
    plot_svm(X_train, y_train, svm_clf, "petal length", "petal width", "Linear SVM")
    print(f"train accuracy: {accuracy_score(y_train, svm_clf.predict(X_train))}")
    print(f"test accuracy: {accuracy_score(y_test, svm_clf.predict(X_test))}")
