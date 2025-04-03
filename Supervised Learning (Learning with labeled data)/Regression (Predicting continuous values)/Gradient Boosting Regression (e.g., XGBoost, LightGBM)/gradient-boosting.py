import numpy as np

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def _mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _mse_gradient(self, y_true, y_pred):
        return -2 * (y_true - y_pred)

    def _build_tree(self, X, y, depth=0):
        m, n = X.shape
        if depth == self.max_depth or m <= 1:
            return np.mean(y)

        best_mse = float('inf')
        best_split = {}

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                left_mean = np.mean(y[left_indices])
                right_mean = np.mean(y[right_indices])

                left_mse = self._mse(y[left_indices], left_mean)
                right_mse = self._mse(y[right_indices], right_mean)

                mse = (len(y[left_indices]) / m) * left_mse + (len(y[right_indices]) / m) * right_mse

                if mse < best_mse:
                    best_mse = mse
                    best_split['feature'] = feature
                    best_split['threshold'] = threshold
                    best_split['left_indices'] = left_indices
                    best_split['right_indices'] = right_indices

        if not best_split:
            return np.mean(y)

        feature = best_split['feature']
        threshold = best_split['threshold']
        left_indices = best_split['left_indices']
        right_indices = best_split['right_indices']

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return {'feature': feature, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def _traverse_tree(self, x, tree):
        if not isinstance(tree, dict): # if tree is a leaf
            return tree

        if x[tree['feature']] <= tree['threshold']:
            return self._traverse_tree(x, tree['left'])
        else:
            return self._traverse_tree(x, tree['right'])

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        predictions = np.full(len(y), self.initial_prediction)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = self._mse_gradient(y, predictions)
            tree = self._build_tree(X, residuals)
            self.trees.append(tree)

            tree_predictions = np.array([self._traverse_tree(x, tree) for x in X])
            predictions += self.learning_rate * tree_predictions

    def predict(self, X):
        predictions = np.full(len(X), self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * np.array([self._traverse_tree(x, tree) for x in X])
        return predictions

# Example Usage:
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gbr.fit(X_train, y_train)
predictions = gbr.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")