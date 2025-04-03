import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _mse(self, y):
        return np.mean((y - np.mean(y))**2)

    def _best_split(self, X, y):
        m, n = X.shape
        best_mse = float('inf')
        best_split = {}

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                mse_left = self._mse(y[left_indices])
                mse_right = self._mse(y[right_indices])
                mse = (len(y[left_indices]) / len(y) * mse_left) + (len(y[right_indices]) / len(y) * mse_right)

                if mse < best_mse:
                    best_mse = mse
                    best_split['feature'] = feature
                    best_split['threshold'] = threshold
                    best_split['left_indices'] = left_indices
                    best_split['right_indices'] = right_indices

        return best_split

    def _build_tree(self, X, y, depth=0):
        m, n = X.shape

        if (depth == self.max_depth or m < self.min_samples_split):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        best_split = self._best_split(X, y)

        if not best_split:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        feature = best_split['feature']
        threshold = best_split['threshold']
        left_indices = best_split['left_indices']
        right_indices = best_split['right_indices']

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature, threshold, left_child, right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

# Example Usage:
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_regressor = DecisionTreeRegressor(max_depth=5)
dt_regressor.fit(X_train, y_train)
predictions = dt_regressor.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")