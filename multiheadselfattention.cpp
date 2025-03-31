#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::inner_product

// Simple matrix multiplication function (you might want a more optimized one)
std::vector<std::vector<float>> multiply_matrices(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    int rows_a = a.size();
    int cols_a = a[0].size();
    int rows_b = b.size();
    int cols_b = b[0].size();

    if (cols_a != rows_b) {
        throw std::runtime_error("Matrices can't be multiplied! Incompatible dimensions.");
    }

    std::vector<std::vector<float>> result(rows_a, std::vector<float>(cols_b, 0.0f));
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            for (int k = 0; k < cols_a; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

// Function to transpose a matrix
std::vector<std::vector<float>> transpose_matrix(const std::vector<std::vector<float>>& matrix) {
    if (matrix.empty()) return {};
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// Function to calculate the dot product of two vectors
float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have the same size for dot product.");
    }
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

// Function to calculate softmax of a vector
std::vector<float> softmax(const std::vector<float>& values) {
    std::vector<float> result(values.size());
    float sum = 0.0f;
    for (float v : values) {
        sum += std::exp(v);
    }
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = std::exp(values[i]) / sum;
    }
    return result;
}

// Scaled dot-product attention for a single head
std::vector<std::vector<float>> scaled_dot_product_attention(
    const std::vector<std::vector<float>>& query,
    const std::vector<std::vector<float>>& key,
    const std::vector<std::vector<float>>& value
) {
    // 1. Calculate attention scores (dot product of query and key)
    std::vector<std::vector<float>> scores = multiply_matrices(query, transpose_matrix(key));

    // 2. Scale the scores
    float d_k = key[0].size(); // Dimension of the key vectors
    for (auto& row : scores) {
        for (float& score : row) {
            score /= std::sqrt(d_k);
        }
    }

    // 3. Apply softmax to get attention weights
    std::vector<std::vector<float>> attention_weights(scores.size());
    for (const auto& row : scores) {
        attention_weights.push_back(softmax(row));
    }

    // 4. Multiply attention weights with value to get the output
    return multiply_matrices(attention_weights, value);
}

// Multi-head self-attention
std::vector<std::vector<float>> multi_head_self_attention(
    const std::vector<std::vector<float>>& input_sequence,
    int num_heads,
    int head_dim // Dimension of each attention head
) {
    int seq_len = input_sequence.size();
    int model_dim = input_sequence[0].size();

    if (model_dim != num_heads * head_dim) {
        throw std::runtime_error("Model dimension must be equal to num_heads * head_dim.");
    }

    std::vector<std::vector<std::vector<float>>> all_head_outputs;

    for (int i = 0; i < num_heads; ++i) {
        // Linear projections for query, key, and value for this head
        std::vector<std::vector<float>> query_weight(model_dim, std::vector<float>(head_dim)); // Initialize randomly
        std::vector<std::vector<float>> key_weight(model_dim, std::vector<float>(head_dim));   // Initialize randomly
        std::vector<std::vector<float>> value_weight(model_dim, std::vector<float>(head_dim)); // Initialize randomly

        std::vector<std::vector<float>> query = multiply_matrices(input_sequence, query_weight);
        std::vector<std::vector<float>> key = multiply_matrices(input_sequence, key_weight);
        std::vector<std::vector<float>> value = multiply_matrices(input_sequence, value_weight);

        // Apply scaled dot-product attention
        all_head_outputs.push_back(scaled_dot_product_attention(query, key, value));
    }

    // Concatenate the outputs of all heads
    std::vector<std::vector<float>> concatenated_output(seq_len, std::vector<float>(num_heads * head_dim));
    for (int i = 0; i < seq_len; ++i) {
        for (int head = 0; head < num_heads; ++head) {
            for (int j = 0; j < head_dim; ++j) {
                concatenated_output[i][head * head_dim + j] = all_head_outputs[head][i][j];
            }
        }
    }

    // Final linear projection
    std::vector<std::vector<float>> output_weight(num_heads * head_dim, std::vector<float>(model_dim)); // Initialize randomly
    return multiply_matrices(concatenated_output, output_weight);
}

int main() {
    // Example usage
    std::vector<std::vector<float>> input_sequence = {
        {1.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 2.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f, 2.0f}
    };

    int num_heads = 2;
    int head_dim = 2;

    std::vector<std::vector<float>> output = multi_head_self_attention(input_sequence, num_heads, head_dim);

    std::cout << "Multi-Head Self-Attention Output:" << std::endl;
    for (const auto& row : output) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}