## Linear Regression (C++)

This section contains a C++ implementation of simple linear regression.

**What is Linear Regression?**

Linear regression is a fundamental statistical and machine learning algorithm used to model the relationship between a dependent variable (the target or output) and one or more independent variables (the features or inputs) by fitting a linear equation to the observed data. The goal is to find the line that best predicts the target variable based on the input variable(s). In our current implementation, we're focusing on *simple* linear regression, which involves only one independent variable.

**C++ Implementation Details**

Our C++ implementation provides a `LinearRegression` class with the following functionalities:

* **`LinearRegression()`:** The constructor initializes the `slope_` and `intercept_` of the linear equation to 0.0.
* **`fit(const std::vector<float>& x, const std::vector<float>& y)`:** This method takes two constant references to vectors of floats: `x` representing the independent variable and `y` representing the dependent variable. It calculates the best-fitting slope and intercept using the Ordinary Least Squares (OLS) method. The formulas used are derived from minimizing the sum of the squared differences between the observed and predicted values. It also includes a basic error check to ensure the input vectors are valid.
* **`predict(float x_new) const`:** This method takes a new value for the independent variable (`x_new`) and returns the predicted value of the dependent variable based on the learned slope and intercept. The prediction is made using the equation: `y = slope_ * x_new + intercept_`.

**How to Use**

To use the `LinearRegression` class, you'll first need to include the necessary headers (`iostream`, `vector`, and `numeric`). Then, you can create an instance of the `LinearRegression` class, provide your training data (features and targets) to the `fit` method, and then use the `predict` method to make predictions on new data points.

Here's a snippet from the `main` function showing an example:

```cpp
// Sample data
std::vector<float> features = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
std::vector<float> targets = {2.0f, 4.0f, 5.0f, 4.0f, 5.0f};

LinearRegression model;
model.fit(features, targets);

std::cout << "Slope (m): " << model.slope_ << std::endl;
std::cout << "Intercept (c): " << model.intercept_ << std::endl;

// Make a prediction
float new_feature = 6.0f;
float prediction = model.predict(new_feature);
std::cout << "Prediction for " << new_feature << ": " << prediction << std::endl;