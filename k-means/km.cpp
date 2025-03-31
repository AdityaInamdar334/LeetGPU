#include <iostream>
#include <vector>
#include <cmath>
#include <limits> // For std::numeric_limits
#include <random>
#include <algorithm> // For std::generate, std::remove_if

// Helper function to calculate Euclidean distance between two data points
float euclidean_distance(const std::vector<float>& p1, const std::vector<float>& p2) {
    float distance = 0.0f;
    for (size_t i = 0; i < p1.size(); ++i) {
        distance += std::pow(p1[i] - p2[i], 2);
    }
    return std::sqrt(distance);
}

class KMeans {
public:
    int k_; // Number of clusters
    int max_iterations_;
    std::vector<std::vector<float>> centroids_;
    std::vector<int> cluster_assignments_;

    KMeans(int k = 3, int max_iters = 100) : k_(k), max_iterations_(max_iters) {}

    void fit(const std::vector<std::vector<float>>& data) {
        if (data.empty() || data.size() < k_) {
            std::cerr << "Error: Not enough data points for the number of clusters." << std::endl;
            return;
        }

        // 1. Initialize centroids randomly
        initialize_centroids(data);

        for (int iteration = 0; iteration < max_iterations_; ++iteration) {
            std::vector<std::vector<float>> new_centroids(k_, std::vector<float>(data[0].size(), 0.0f));
            std::vector<int> new_cluster_assignments(data.size());
            std::vector<int> cluster_counts(k_, 0);
            bool centroids_changed = false;

            // 2. Assign each data point to the nearest centroid
            for (size_t i = 0; i < data.size(); ++i) {
                int best_centroid = -1;
                float min_distance = std::numeric_limits<float>::max();

                for (int j = 0; j < k_; ++j) {
                    float distance = euclidean_distance(data[i], centroids_[j]);
                    if (distance < min_distance) {
                        min_distance = distance;
                        best_centroid = j;
                    }
                }
                new_cluster_assignments[i] = best_centroid;
                for (size_t j = 0; j < data[i].size(); ++j) {
                    new_centroids[best_centroid][j] += data[i][j];
                }
                cluster_counts[best_centroid]++;
            }

            // 3. Update centroids
            for (int i = 0; i < k_; ++i) {
                if (cluster_counts[i] > 0) {
                    for (size_t j = 0; j < new_centroids[i].size(); ++j) {
                        new_centroids[i][j] /= cluster_counts[i];
                    }
                } else {
                    // Handle empty clusters by re-initializing the centroid
                    new_centroids[i] = data[std::rand() % data.size()];
                }
                if (new_centroids[i] != centroids_[i]) {
                    centroids_changed = true;
                }
            }

            cluster_assignments_ = new_cluster_assignments;
            centroids_ = new_centroids;

            // 4. Check for convergence
            if (!centroids_changed) {
                std::cout << "Converged at iteration " << iteration + 1 << std::endl;
                break;
            }
            if (iteration == max_iterations_ - 1) {
                std::cout << "Reached maximum iterations." << std::endl;
            }
        }
    }

    const std::vector<int>& get_cluster_assignments() const {
        return cluster_assignments_;
    }

    const std::vector<std::vector<float>>& get_centroids() const {
        return centroids_;
    }

private:
    void initialize_centroids(const std::vector<std::vector<float>>& data) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, data.size() - 1);

        centroids_.resize(k_);
        std::vector<int> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        for (int i = 0; i < k_; ++i) {
            centroids_[i] = data[indices[i]];
        }
    }
};

int main() {
    // Sample data
    std::vector<std::vector<float>> data = {
        {1.0f, 1.0f},
        {1.5f, 1.5f},
        {5.0f, 5.0f},
        {5.5f, 5.5f},
        {3.0f, 3.0f},
        {4.0f, 4.0f}
    };

    int k = 2; // Number of clusters
    KMeans kmeans(k, 50);
    kmeans.fit(data);

    std::cout << "Cluster Assignments:" << std::endl;
    const auto& assignments = kmeans.get_cluster_assignments();
    for (int assignment : assignments) {
        std::cout << assignment << " ";
    }
    std::cout << std::endl;

    std::cout << "Centroids:" << std::endl;
    const auto& centroids = kmeans.get_centroids();
    for (const auto& centroid : centroids) {
        std::cout << "[";
        for (float val : centroid) {
            std::cout << val << " ";
        }
        std::cout << "] ";
    }
    std::cout << std::endl;

    return 0;
}