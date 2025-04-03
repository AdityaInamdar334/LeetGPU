
# Naive Bayes Classifier Implementation

This is a basic implementation of the **Naive Bayes classifier** using Python and the `scikit-learn` library. The implementation demonstrates how to train a Naive Bayes model on a dataset, make predictions, and evaluate the model's performance.

## Features

- Multiclass classification using the Iris dataset.
- Data preprocessing with feature scaling.
- Model evaluation with accuracy and confusion matrix.

## Requirements

- Python 3.x
- scikit-learn
- numpy

You can install the necessary dependencies using `pip`:

```bash
pip install numpy scikit-learn
```

## Dataset

This code uses the **Iris** dataset, which is a built-in dataset in `scikit-learn`. It contains three classes of iris flowers, and the goal is to predict the class of a given flower based on its features.

## Steps

1. **Load the Dataset**: The Iris dataset is loaded, and the goal is to classify the flower species into one of three classes.
2. **Preprocessing**: The features are scaled using `StandardScaler` to improve the model's performance.
3. **Model Training**: The Naive Bayes model (`GaussianNB`) is created and trained on the training data.
4. **Evaluation**: The model is evaluated using accuracy and confusion matrix metrics to assess its performance.

## Code

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
```

## Output

- **Accuracy**: The model's accuracy on the test dataset.
- **Confusion Matrix**: A matrix showing the model's classification results, which helps in understanding how many true positives, false positives, true negatives, and false negatives were predicted.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
