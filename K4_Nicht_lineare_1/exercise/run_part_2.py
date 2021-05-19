from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, "..")
from exercise.utility import plot_cluster, plot_svm


def get_svm_clf(kernel: str):
    """Returns an already defined SVM to use for classification with different kernels"""
    if kernel == "linear":
        svm_clf = SVC(C=100, kernel="linear", random_state=42, class_weight="balanced")
    elif kernel == "rbf":
        svm_clf = SVC(C=100, kernel="rbf", random_state=42)
    elif kernel == "poly":
        svm_clf = SVC(C=100, kernel="poly", random_state=42, gamma="auto", degree=2)
    else:
        raise Exception("No such Kernel")
    return svm_clf


def optional_2():
    pass


def question_3():
    """Returns the type of kernel ∈ ["linear", "poly", "rbf"], for which the training score is best."""
    return ""


def question_4():
    """Returns the type of kernel ∈ ["linear", "poly", "rbf"], for which the test score is best."""
    return ""


if __name__ == "__main__":
    # load data set
    iris = load_iris()
    # now we look at all classes, but only at features 2, 3
    X = iris["data"][:, (2, 3)]
    # scale data. SVMs are sensitive to varying feature scales
    X = StandardScaler().fit_transform(X)
    # SVMs are binary classifiers. Thus, they can only distinguish between class X and not class X.
    # Since our target data contains the class labels for each data point, we need to redefine it as:
    # 1 if the data point is of class x, 0 if the data point is not of class x
    # in this case the class x is iris versicolor with the label 1
    y = (iris["target"] == 1).astype(float)
    plot_cluster(X, y, "petal length", "petal width", "Versicolor (Brown) and not Versicolor (Blue)")
    # split data in to train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Create and train SVM
    svm_clf = get_svm_clf("linear")
    svm_clf.fit(X_train, y_train)
    # plot decision boundary
    plot_svm(X_train, y_train, svm_clf, "petal length", "petal width", f"{svm_clf.kernel}-SVM")
    print(f"train accuracy: {accuracy_score(y_train, svm_clf.predict(X_train))}")
    print(f"test accuracy: {accuracy_score(y_test, svm_clf.predict(X_test))}")
