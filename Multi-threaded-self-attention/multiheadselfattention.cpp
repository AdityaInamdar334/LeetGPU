#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::inner_product
#include <random> // For initializing weights (for demonstration)

// Helper function for matrix multiplication
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

// Helper function to transpose a matrix
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

// Helper function for softmax
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
    const std::vector<std::vector<float>>& query, // Shape: (seq_len, head_dim)
    const std::vector<std::vector<float>>& key,   // Shape: (seq_len, head_dim)
    const std::vector<std::vector<float>>& value  // Shape: (seq_len, head_dim)
) {
    // 1. Calculate attention scores: dot product of query and key (transposed)
    // scores = query * key^T. Shape: (seq_len, head_dim) * (head_dim, seq_len) = (seq_len, seq_len)
    std::vector<std::vector<float>> scores = multiply_matrices(query, transpose_matrix(key));

    // 2. Scale the scores by the square root of the key dimension
    float d_k = key[0].size();
    for (auto& row : scores) {
        for (float& score : row) {
            score /= std::sqrt(d_k);
        }
    }

    // 3. Apply softmax to get attention weights
    std::vector<std::vector<float>> attention_weights(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        attention_weights[i] = softmax(scores[i]);
    }

    // 4. Multiply attention weights with value to get the output for this head
    // output = attention_weights * value. Shape: (seq_len, seq_len) * (seq_len, head_dim) = (seq_len, head_dim)
    return multiply_matrices(attention_weights, value);
}

// Multi-head self-attention
std::vector<std::vector<float>> multi_head_self_attention(
    const std::vector<std::vector<float>>& input_sequence, // Shape: (seq_len, model_dim)
    int num_heads,
    int head_dim // Dimension of each attention head
) {
    int seq_len = input_sequence.size();
    int model_dim = input_sequence[0].size();

    if (model_dim != num_heads * head_dim) {
        throw std::runtime_error("Model dimension must be equal to num_heads * head_dim.");
    }

    std::vector<std::vector<std::vector<float>>> all_head_outputs;

    // Initialize random number generator for weights (for demonstration)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-1.0, 1.0);

    for (int i = 0; i < num_heads; ++i) {
        // Linear projections for query, key, and value for this head
        std::vector<std::vector<float>> query_weight(model_dim, std::vector<float>(head_dim));
        std::vector<std::vector<float>> key_weight(model_dim, std::vector<float>(head_dim));
        std::vector<std::vector<float>> value_weight(model_dim, std::vector<float>(head_dim));

        // Initialize weights randomly (for demonstration purposes)
        for (auto& row : query_weight) for (float& val : row) val = distrib(gen);
        for (auto& row : key_weight) for (float& val : row) val = distrib(gen);
        for (auto& row : value_weight) for (float& val : row) val = distrib(gen);

        // Calculate query, key, and value for this head
        // Shape: (seq_len, model_dim) * (model_dim, head_dim) = (seq_len, head_dim)
        std::vector<std::vector<float>> query = multiply_matrices(input_sequence, query_weight);
        std::vector<std::vector<float>> key = multiply_matrices(input_sequence, key_weight);
        std::vector<std::vector<float>> value = multiply_matrices(input_sequence, value_weight);

        // Apply scaled dot-product attention for this head
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

    // Final linear projection (optional, but often included)
    std::vector<std::vector<float>> output_weight(num_heads * head_dim, std::vector<float>(model_dim));
    for (auto& row : output_weight) for (float& val : row) val = distrib(gen); // Initialize randomly
    return multiply_matrices(concatenated_output, output_weight);
}

int main() {
    // Example usage
    int seq_len = 3;
    int model_dim = 4;
    std::vector<std::vector<float>> input_sequence(seq_len, std::vector<float>(model_dim));
    // Initialize input sequence with some values
    input_sequence[0] = {1.0f, 0.0f, 1.0f, 0.0f};
    input_sequence[1] = {0.0f, 2.0f, 0.0f, 1.0f};
    input_sequence[2] = {1.0f, 1.0f, 0.0f, 2.0f};

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