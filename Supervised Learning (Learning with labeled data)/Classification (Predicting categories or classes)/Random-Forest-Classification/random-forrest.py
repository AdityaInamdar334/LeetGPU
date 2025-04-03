import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _best_split(self, X, y):
        m, n = X.shape
        if self.n_features is None:
            self.n_features = int(np.sqrt(n))

        best_entropy = float('inf')
        best_split = {}

        features = np.random.choice(n, self.n_features, replace=False)

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                entropy_left = self._entropy(y[left_indices])
                entropy_right = self._entropy(y[right_indices])
                entropy = (len(y[left_indices]) / len(y) * entropy_left) + (len(y[right_indices]) / len(y) * entropy_right)

                if entropy < best_entropy:
                    best_entropy = entropy
                    best_split['feature'] = feature
                    best_split['threshold'] = threshold
                    best_split['left_indices'] = left_indices
                    best_split['right_indices'] = right_indices

        return best_split

    def _build_tree(self, X, y, depth=0):
        m, n = X.shape
        n_labels = len(np.unique(y))

        if (depth == self.max_depth or n_labels == 1 or m < self.min_samples_split):
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)

        best_split = self._best_split(X, y)
        if not best_split:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        feature = best_split['feature']
        threshold = best_split['threshold']
        left_indices = best_split['left_indices']
        right_indices = best_split['right_indices']

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature, threshold, left_child, right_child)

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), len(X), replace=True) #bootstraping
            self.trees.append(self._build_tree(X[indices], y[indices]))

    def _traverse_tree(self, x, tree):
        if tree.value is not None:
            return tree.value

        if x[tree.feature] <= tree.threshold:
            return self._traverse_tree(x, tree.left)
        return self._traverse_tree(x, tree.right)

    def predict(self, X):
        predictions = []
        for x in X:
            tree_predictions = [self._traverse_tree(x, tree) for tree in self.trees]
            predictions.append(Counter(tree_predictions).most_common(1)[0][0])
        return np.array(predictions)

# Example Usage:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForest(n_trees=10, max_depth=10)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, predictions)}")