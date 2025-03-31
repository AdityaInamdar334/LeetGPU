#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::inner_product
#include <random>

// Helper function for matrix-vector multiplication
std::vector<float> multiply_matrix_vector(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vector) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    int vec_size = vector.size();

    if (cols != vec_size) {
        throw std::runtime_error("Matrix and vector dimensions do not match for multiplication.");
    }

    std::vector<float> result(rows, 0.0f);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

// Helper function for vector addition
std::vector<float> add_vectors(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have the same size for addition.");
    }
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// Helper function for tanh activation
std::vector<float> tanh_activation(const std::vector<float>& values) {
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = std::tanh(values[i]);
    }
    return result;
}

class SimpleRNN {
public:
    int input_size_;
    int hidden_size_;
    std::vector<std::vector<float>> weight_ih_; // Input to hidden weights
    std::vector<std::vector<float>> weight_hh_; // Hidden to hidden weights
    std::vector<float> bias_h_;

    SimpleRNN(int input_size, int hidden_size) : input_size_(input_size), hidden_size_(hidden_size) {
        // Initialize weights and biases randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(-1.0, 1.0);

        weight_ih_.resize(hidden_size_, std::vector<float>(input_size_));
        weight_hh_.resize(hidden_size_, std::vector<float>(hidden_size_));
        bias_h_.resize(hidden_size_);

        for (auto& row : weight_ih_) for (float& val : row) val = distrib(gen);
        for (auto& row : weight_hh_) for (float& val : row) val = distrib(gen);
        for (float& val : bias_h_) val = distrib(gen);
    }

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input_sequence) {
        std::vector<std::vector<float>> hidden_states;
        std::vector<float> previous_hidden_state(hidden_size_, 0.0f); // Initialize hidden state to zeros

        for (const auto& input_t : input_sequence) {
            // 1. Input to hidden: W_ih * x_t
            std::vector<float> input_hidden = multiply_matrix_vector(weight_ih_, input_t);

            // 2. Hidden to hidden: W_hh * h_{t-1}
            std::vector<float> hidden_hidden = multiply_matrix_vector(weight_hh_, previous_hidden_state);

            // 3. Add them up and the bias: W_ih * x_t + W_hh * h_{t-1} + b_h
            std::vector<float> pre_activation = add_vectors(add_vectors(input_hidden, hidden_hidden), bias_h_);

            // 4. Apply tanh activation
            std::vector<float> current_hidden_state = tanh_activation(pre_activation);

            hidden_states.push_back(current_hidden_state);
            previous_hidden_state = current_hidden_state; // Update for the next time step
        }

        return hidden_states;
    }
};

int main() {
    // Example usage:
    int input_size = 3; // Number of features at each time step
    int hidden_size = 2; // Number of neurons in the hidden layer
    int sequence_length = 4;

    // Create a sample input sequence
    std::vector<std::vector<float>> input_sequence(sequence_length, std::vector<float>(input_size));
    input_sequence[0] = {1.0f, 0.0f, -1.0f};
    input_sequence[1] = {0.5f, 0.5f, 0.0f};
    input_sequence[2] = {0.0f, -0.5f, 1.0f};
    input_sequence[3] = {-1.0f, 0.0f, 0.5f};

    // Create an RNN instance
    SimpleRNN rnn(input_size, hidden_size);

    // Perform the forward pass
    std::vector<std::vector<float>> hidden_states = rnn.forward(input_sequence);

    // Print the hidden states at each time step
    std::cout << "Hidden States:" << std::endl;
    for (size_t i = 0; i < hidden_states.size(); ++i) {
        std::cout << "Time step " << i << ": [";
        for (float val : hidden_states[i]) {
            std::cout << val << " ";
        }
        std::cout << "]" << std::endl;
    }

    return 0;
}