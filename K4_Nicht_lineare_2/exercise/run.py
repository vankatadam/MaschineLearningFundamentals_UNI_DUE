import sys
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from exercise.cart import Tree


# load data
data = load_iris()
X = data.data[:,:2]
y = data.target


# train decision tree
tree = Tree(max_depth=2)
tree.fit(X, y)

print("\n-------------------------")
print("Trained Decision Tree: ")
print("-------------------------")
tree.print_tree(tree.root)
print("-------------------------\n\n")


# measure accuracy
accuracy = 0.
for i, row in enumerate(X):
    if y[i] == tree.predict(X[i]):
        accuracy += 1
accuracy /= len(X)
print("Accuracy: %.2f\n" %(accuracy))
