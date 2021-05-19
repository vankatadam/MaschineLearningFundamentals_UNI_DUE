import numpy as np


class Tree:
    
    def __init__(self, max_depth=2, min_samples_split=2):
        self.root = None
        self.max_depth=max_depth
        self.min_samples_split = min_samples_split


    def evaluate_split(self, feature_idx: int, threshold: float, group):
        """
        Iterate through the group of samples and decide whether it belongs to the left or right bucket.
        """
        left, right = list(), list()
        # TODO
        return left, right


    def calc_gini_idx(self, groups, classes):
        """
        Calculate the gini index for all groups.

        Parameters
        ----------
        groups : list
            List of ndarrays. Each group in groups contains all samples for a single node.
            Each group is structured as samples x features. The last feature represents the class label.

        classes : list
            List of class values. For a dataset with 3 classes, classes has a length of 3.
        """
        gini = 0.0
        # TODO
        return gini


    def find_best_split(self, dataset):
        """
        Calculate the optimal split for all possible splits. We consider each feature value as a split candidate.

        Parameters
        ----------
        dataset : ndarray
            Dataset is structured as samples x features. The last feature represents the class label.

        return :
            groups : tuple
                Tuple of two lists
                
        """
        # best index, value, score and data groups
        b_index, b_value, b_score, b_groups = np.Inf, np.Inf, np.Inf, ([], [])
        # TODO
        return {'index':b_index, 'value':b_value, 'groups':b_groups}


    def create_leaf(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)


    def split(self, node, depth, max_depth, min_split_size):
        left, right = node['groups']
        del(node['groups'])
        
        if not left or not right:
            node['left'] = node['right'] = self.create_leaf(left + right)
            return
        
        if depth >= max_depth:
            node['left']  = self.create_leaf(left)
            node['right'] = self.create_leaf(right)
            return

        if len(left) <= min_split_size:
            node['left'] = self.create_leaf(left)
        else:
            node['left'] = self.find_best_split(left)
            self.split(node['left'], depth+1, max_depth, min_split_size)

        if len(right) <= min_split_size:
            node['right'] = self.create_leaf(right)
        else:
            node['right'] = self.find_best_split(right)
            self.split(node['right'], depth+1, max_depth, min_split_size)


    def fit(self, X, y):
        """
        Trains the decision tree.
        """
        self.X = X
        self.y = y
        self.root = self.find_best_split(np.concatenate((X, y.reshape(-1,1)), axis=1))
        self.split(self.root, 1, self.max_depth, self.min_samples_split)
        return self


    def predict(self, sample, node=None):
        """
        Returns the predicted class for the given sample.
        """
        if not node:
            node = self.root
        if sample[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(sample, node=node['left'])
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(sample, node=node['right'])
            else:
                return node['right']


    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))

