import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a toy dataset for binary classification
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert the dataset into DMatrix, which is the internal data structure that XGBoost uses
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the parameters for the model
params = {
    'objective': 'binary:logistic',  # Binary classification
    'max_depth': 3,                  # Maximum depth of a tree
    'eta': 0.1,                      # Learning rate
    'eval_metric': 'logloss'         # Evaluation metric
}

# Train the model using the training data
num_round = 100  # Number of boosting rounds (iterations)
bst = xgb.train(params, dtrain, num_round)

# Make predictions on the test data
preds = bst.predict(dtest)
# Convert predictions to binary (0 or 1) using a threshold of 0.5
preds_binary = [1 if p > 0.5 else 0 for p in preds]

# Calculate accuracy
accuracy = accuracy_score(y_test, preds_binary)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Optionally, save the model for future use
bst.save_model('xgb_model.json')

# Optionally, load the model later
# bst_loaded = xgb.Booster()
# bst_loaded.load_model('xgb_model.json')
