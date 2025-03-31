#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>

// Structure to represent a node in the decision tree
struct TreeNode {
    int feature_index; // Index of the feature to split on
    float threshold;     // Threshold value for the split
    int predicted_class; // Class label for leaf nodes (-1 if not a leaf)
    TreeNode* left;
    TreeNode* right;

    TreeNode() : feature_index(-1), threshold(0.0f), predicted_class(-1), left(nullptr), right(nullptr) {}
};

std::pair<int, float> find_best_split(const std::vector<std::vector<float>>& features, const std::vector<int>& labels) {
    int best_feature_index = -1;
    float best_threshold = -1.0f;
    // Simplistic split: just pick the first feature and a midpoint

    if (!features.empty() && !features[0].empty()) {
        best_feature_index = 0;
        float min_val = features[0][0];
        float max_val = features[0][0];
        for (const auto& feature_vec : features) {
            min_val = std::min(min_val, feature_vec[0]);
            max_val = std::max(max_val, feature_vec[0]);
        }
        best_threshold = (min_val + max_val) / 2.0f;
    }
    return {best_feature_index, best_threshold};
}

TreeNode* build_tree(const std::vector<std::vector<float>>& features, const std::vector<int>& labels, int depth, int max_depth) {
    if (labels.empty() || depth >= max_depth) {
        TreeNode* leaf_node = new TreeNode();
        std::map<int, int> class_counts;
        for (int label : labels) {
            class_counts[label]++;
        }
        int majority_class = -1;
        int max_count = -1;
        for (const auto& pair : class_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                majority_class = pair.first;
            }
        }
        leaf_node->predicted_class = majority_class;
        return leaf_node;
    }

    std::pair<int, float> best_split = find_best_split(features, labels);
    int split_feature_index = best_split.first;
    float split_threshold = best_split.second;

    if (split_feature_index == -1) {
        TreeNode* leaf_node = new TreeNode();
        std::map<int, int> class_counts;
        for (int label : labels) {
            class_counts[label]++;
        }
        int majority_class = -1;
        int max_count = -1;
        for (const auto& pair : class_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                majority_class = pair.first;
            }
        }
        leaf_node->predicted_class = majority_class;
        return leaf_node;
    }

    std::vector<std::vector<float>> left_features, right_features;
    std::vector<int> left_labels, right_labels;
    for (size_t i = 0; i < features.size(); ++i) {
        if (features[i][split_feature_index] < split_threshold) {
            left_features.push_back(features[i]);
            left_labels.push_back(labels[i]);
        } else {
            right_features.push_back(features[i]);
            right_labels.push_back(labels[i]);
        }
    }

    TreeNode* current_node = new TreeNode();
    current_node->feature_index = split_feature_index;
    current_node->threshold = split_threshold;
    current_node->left = build_tree(left_features, left_labels, depth + 1, max_depth);
    current_node->right = build_tree(right_features, right_labels, depth + 1, max_depth);

    return current_node;
}

// Removed const here!
int predict(TreeNode* node, const std::vector<float>& sample) {
    if (node->predicted_class != -1) {
        return node->predicted_class;
    }

    if (sample[node->feature_index] < node->threshold) {
        return predict(node->left, sample);
    } else {
        return predict(node->right, sample);
    }
}

class DecisionTreeClassifier {
public:
    TreeNode* root_;
    int max_depth_;

    DecisionTreeClassifier(int max_depth = 5) : root_(nullptr), max_depth_(max_depth) {}

    void fit(const std::vector<std::vector<float>>& features, const std::vector<int>& labels) {
        root_ = build_tree(features, labels, 0, max_depth_);
    }

    int predict(const std::vector<float>& sample) const {
        if (root_ == nullptr) {
            throw std::runtime_error("Model not fitted yet.");
        }
        return predict(root_, sample);
    }

private:
    TreeNode* build_tree(const std::vector<std::vector<float>>& features, const std::vector<int>& labels, int depth, int max_depth);
    std::pair<int, float> find_best_split(const std::vector<std::vector<float>>& features, const std::vector<int>& labels);
    // Kept const here, as this is a member function of a const object
    int predict(TreeNode* node, const std::vector<float>& sample) const;
};

// Move the implementations here to avoid redefinition errors
TreeNode* DecisionTreeClassifier::build_tree(const std::vector<std::vector<float>>& features, const std::vector<int>& labels, int depth, int max_depth) {
    return ::build_tree(features, labels, depth, max_depth);
}

std::pair<int, float> DecisionTreeClassifier::find_best_split(const std::vector<std::vector<float>>& features, const std::vector<int>& labels) {
    return ::find_best_split(features, labels);
}

// Kept const here!
int DecisionTreeClassifier::predict(TreeNode* node, const std::vector<float>& sample) const {
    return ::predict(node, sample);
}

int main() {
    // Simple example data (2 features, 2 classes)
    std::vector<std::vector<float>> features = {
        {2.0f, 2.0f},
        {2.0f, 3.0f},
        {1.0f, 2.0f},
        {3.0f, 1.0f},
        {4.0f, 4.0f}
    };
    std::vector<int> labels = {0, 0, 1, 1, 0};

    DecisionTreeClassifier classifier(3); // Set a maximum depth
    classifier.fit(features, labels);

    // Predict for a new sample
    std::vector<float> sample = {2.5f, 2.5f};
    int prediction = classifier.predict(sample);
    std::cout << "Prediction for sample [2.5, 2.5]: " << prediction << std::endl;

    std::vector<float> sample2 = {1.5f, 2.1f};
    int prediction2 = classifier.predict(sample2);
    std::cout << "Prediction for sample [1.5, 2.1]: " << prediction2 << std::endl;

    return 0;
}