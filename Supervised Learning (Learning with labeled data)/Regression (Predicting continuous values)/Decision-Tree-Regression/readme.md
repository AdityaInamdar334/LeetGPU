
# Simple Decision Tree Regressor

This repository contains a basic implementation of a Decision Tree Regressor in Python. It's designed to provide a clear understanding of how decision trees can be used for regression tasks.

## Features

* **Basic Decision Tree Regression:** Implements the core logic of a decision tree regressor.
* **Mean Squared Error (MSE):** Uses MSE to determine the best splits in the tree.
* **Mean-Based Leaf Values:** Leaf nodes store the mean of the target values.
* **Clear and Commented Code:** Designed for readability and educational purposes.
* **Example Usage:** Includes an example using `make_regression` from scikit-learn.

## Requirements

* Python 3.x
* NumPy
* scikit-learn (for the example usage)

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your-username/simple-decision-tree-regressor.git](https://www.google.com/search?q=https://www.google.com/search%3Fq%3Dhttps://github.com/your-username/simple-decision-tree-regressor.git)
    cd simple-decision-tree-regressor
    ```

2.  Install the required packages:

    ```bash
    pip install numpy scikit-learn
    ```

## Usage

1.  Run the provided example:

    ```bash
    python decision_tree_regressor.py
    ```

    This will train a decision tree regressor on a generated regression dataset and print the mean squared error.

2.  You can also integrate the `DecisionTreeRegressor` class into your own projects:

    ```python
    from decision_tree_regressor import DecisionTreeRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Generate and split data
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the decision tree regressor
    dt_regressor = DecisionTreeRegressor(max_depth=5)
    dt_regressor.fit(X_train, y_train)

    # Make predictions
    predictions = dt_regressor.predict(X_test)

    # Evaluate mean squared error
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    

## File Structure

```
simple-decision-tree-regressor/
├── decision_tree_regressor.py   # Implementation of the DecisionTreeRegressor class
└── README.md


## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
