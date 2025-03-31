# Deep Learning and Machine Learning Algorithms

Welcome to this here repository! We're building a collection of implementations for various deep learning and machine learning algorithms in different programming languages. Our main focus right now is C++, but we plan to add Python and maybe even other languages down the line.

## Languages

Currently, you'll find implementations primarily in:

* **C++:** For performance-oriented implementations and a deeper understanding of the algorithms. This includes some CUDA-accelerated code.
* **Python:** We'll be adding Python implementations for easier experimentation and integration with popular ML libraries.
* **[Other languages might be added later]**

## Algorithms Implemented

Here's a list of the algorithms that have been implemented so far, organized by category:

### Supervised Learning

* **Regression:**
    * **Linear Regression (C++)**
* **Classification:**
    * **Logistic Regression (Directory created)**
    * **K-Nearest Neighbors (KNN) (Directory created)**
    * **Support Vector Machines (SVM) (Directory created)**
    * **Naive Bayes (Directory created)**
    * **Decision Tree Classification (Directory created)**
    * **Random Forest Classification (Directory created)**
    * **Gradient Boosting Classification (e.g., XGBoost, LightGBM) (Directory created)**

### Unsupervised Learning

* **Clustering:**
    * **K-Means Clustering (C++)**
    * **Hierarchical Clustering (Directory created)**
    * **DBSCAN (Directory created)**
* **Dimensionality Reduction:**
    * **Principal Component Analysis (PCA) (Directory created)**
    * **t-distributed Stochastic Neighbor Embedding (t-SNE) (Directory created)**

### Neural Networks

* **Core Layers/Functions:**
    * **ReLU Activation Function (C++)**
    * **Matrix Transpose (C++ CUDA)** - This implementation transposes a matrix of 32-bit floating-point numbers on a GPU. See the `Matrix-Transpose` directory.
    * **Multi-Head Self-Attention (C++)**
* **Simple Networks:**
    * **Simple Convolutional Neural Network (CNN) (C++)** - A basic CNN with a single convolutional layer and ReLU activation.
    * **Recurrent Neural Networks (RNNs) - Basic Implementations (C++)** - A simple RNN implementation demonstrating the core concepts.

## Directory Structure

The repository is organized as follows:
