#include <iostream>
#include <vector>
#include <algorithm> // For std::max

std::vector<float> relu_activation(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
    return output;
}

int main() {
    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> output = relu_activation(input);

    std::cout << "Input: ";
    for (float val : input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "ReLU Output: ";
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}