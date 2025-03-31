#include <iostream>
#include <vector>
#include <numeric> // For std::accumulate

class LinearRegression {
public:
    float slope_;
    float intercept_;

    LinearRegression() : slope_(0.0f), intercept_(0.0f) {}

    void fit(const std::vector<float>& x, const std::vector<float>& y) {
        if (x.size() != y.size() || x.empty()) {
            std::cerr << "Error: Input vectors must have the same non-zero size." << std::endl;
            return;
        }

        int n = x.size();
        float sum_x = std::accumulate(x.begin(), x.end(), 0.0f);
        float sum_y = std::accumulate(y.begin(), y.end(), 0.0f);
        float sum_xy = 0.0f;
        float sum_x_squared = 0.0f;

        for (int i = 0; i < n; ++i) {
            sum_xy += x[i] * y[i];
            sum_x_squared += x[i] * x[i];
        }

        float numerator = n * sum_xy - sum_x * sum_y;
        float denominator = n * sum_x_squared - sum_x * sum_x;

        if (denominator == 0) {
            std::cerr << "Error: Denominator is zero. Cannot calculate slope." << std::endl;
            return;
        }

        slope_ = numerator / denominator;
        intercept_ = (sum_y - slope_ * sum_x) / n;
    }

    float predict(float x_new) const {
        return slope_ * x_new + intercept_;
    }
};

int main() {
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

    return 0;
}