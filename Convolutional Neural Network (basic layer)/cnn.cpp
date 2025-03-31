#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::inner_product

// Helper function for 2D convolution
std::vector<std::vector<float>> convolve2D(
    const std::vector<std::vector<float>>& input,
    const std::vector<std::vector<float>>& kernel
) {
    int input_rows = input.size();
    int input_cols = input[0].size();
    int kernel_rows = kernel.size();
    int kernel_cols = kernel[0].size();

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    if (output_rows <= 0 || output_cols <= 0) {
        return {}; // Or handle error appropriately
    }

    std::vector<std::vector<float>> output(output_rows, std::vector<float>(output_cols, 0.0f));

    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < kernel_rows; ++m) {
                for (int n = 0; n < kernel_cols; ++n) {
                    sum += input[i + m][j + n] * kernel[m][n];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

// ReLU activation function (reusing our earlier implementation)
std::vector<std::vector<float>> relu_activation_2d(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> output(input.size(), std::vector<float>(input[0].size()));
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j] = std::max(0.0f, input[i][j]);
        }
    }
    return output;
}

class SimpleCNN {
public:
    std::vector<std::vector<float>> conv_kernel_;

    SimpleCNN(const std::vector<std::vector<float>>& kernel) : conv_kernel_(kernel) {}

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input_image) {
        // 1. Convolutional layer
        std::vector<std::vector<float>> conv_output = convolve2D(input_image, conv_kernel_);

        // 2. ReLU activation
        std::vector<std::vector<float>> relu_output = relu_activation_2d(conv_output);

        return relu_output;
    }
};

int main() {
    // Example input image (a simple 5x5 "image")
    std::vector<std::vector<float>> input_image = {
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 1, 2, 1, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 0, 0, 0}
    };

    // Example convolutional kernel (a simple 3x3 filter)
    std::vector<std::vector<float>> kernel = {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };

    // Create a SimpleCNN object with our kernel
    SimpleCNN cnn(kernel);

    // Perform the forward pass
    std::vector<std::vector<float>> output_feature_map = cnn.forward(input_image);

    // Print the output feature map
    std::cout << "Input Image:" << std::endl;
    for (const auto& row : input_image) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nConvolutional Kernel:" << std::endl;
    for (const auto& row : kernel) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nOutput Feature Map (after convolution and ReLU):" << std::endl;
    for (const auto& row : output_feature_map) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}