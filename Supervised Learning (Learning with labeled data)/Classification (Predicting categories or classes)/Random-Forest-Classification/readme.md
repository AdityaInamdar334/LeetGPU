
# Simple Random Forest Implementation

This repository contains a simple implementation of the Random Forest algorithm in Python. It's designed for educational purposes and provides a clear, step-by-step approach to understanding how random forests work.

## Features

* **Basic Random Forest:** Implements the core logic of a random forest classifier.
* **Bootstrapping:** Uses bootstrapping to create diverse training sets for each tree.
* **Random Feature Selection:** Selects random subsets of features for each split.
* **Entropy-Based Splits:** Uses entropy to determine the best splits in the decision trees.
* **Prediction Aggregation:** Aggregates predictions from all trees using majority voting.
* **Clear and Commented Code:** Designed for readability and understanding.
* **Example Usage:** Includes an example using the Iris dataset.

## Requirements

* Python 3.x
* NumPy
* scikit-learn (for the example usage)

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your-username/simple-random-forest.git](https://www.google.com/search?q=https://github.com/your-username/simple-random-forest.git)
    cd simple-random-forest
    ```

2.  Install the required packages:

    ```bash
    pip install numpy scikit-learn
    ```

## Usage

1.  Run the provided example:

    ```bash
    python random_forest.py
    ```

    This will train a random forest on the Iris dataset and print the accuracy.

2.  You can also integrate the `RandomForest` class into your own projects:

    ```python
    from random_forest import RandomForest
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load and split data
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the random forest
    rf = RandomForest(n_trees=10, max_depth=10)
    rf.fit(X_train, y_train)

    # Make predictions
    predictions = rf.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    
## File Structure


simple-random-forest/
├── random_forest.py   # Implementation of the Random Forest class
└── README.md


## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

