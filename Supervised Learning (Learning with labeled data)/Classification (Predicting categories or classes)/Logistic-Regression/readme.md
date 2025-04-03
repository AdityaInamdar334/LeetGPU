


# Logistic Regression Implementation

This is a basic implementation of **Logistic Regression** using Python and the `scikit-learn` library. The implementation demonstrates how to train a logistic regression model on a dataset, make predictions, and evaluate the model's performance.

## Features

- Binary classification (Setosa vs Non-setosa) using the Iris dataset.
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

The code uses the **Iris** dataset, which is a built-in dataset available in `scikit-learn`. In this implementation, the task is simplified to a binary classification problem: predicting whether the flower is of the **Setosa** species or not.

## Steps

1. **Load the Dataset**: We load the Iris dataset and convert it into a binary classification problem.
2. **Preprocessing**: We scale the features using `StandardScaler` for better performance with logistic regression.
3. **Model Training**: We create a logistic regression model and train it on the training data.
4. **Evaluation**: The model is evaluated using accuracy and confusion matrix to check its performance.

## Code

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = (data.target == 0).astype(int)  # Convert to binary classification: Setosa vs Non-setosa

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression()
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



