#include <iostream>
#include <vector>
#include <xgboost/c_api.h>

// Function to create DMatrix from data
DMatrixHandle createDMatrix(const std::vector<float>& data, const std::vector<float>& labels, int rows, int cols) {
    DMatrixHandle dmat;
    XGDMatrixCreateFromMat(data.data(), rows, cols, -1, &dmat);
    XGDMatrixSetFloatInfo(dmat, "label", labels.data(), labels.size());
    return dmat;
}

int main() {
    // Sample training data (4 samples, 3 features each)
    std::vector<float> train_data = {1.0, 2.0, 3.0,  
                                      4.0, 5.0, 6.0,
                                      7.0, 8.0, 9.0,
                                      10.0, 11.0, 12.0};
    std::vector<float> train_labels = {0, 1, 0, 1};
    int rows = 4, cols = 3;

    // Create training DMatrix
    DMatrixHandle dtrain = createDMatrix(train_data, train_labels, rows, cols);

    // Parameters for XGBoost
    BoosterHandle booster;
    XGBoosterCreate(&dtrain, 1, &booster);
    XGBoosterSetParam(booster, "booster", "gbtree");
    XGBoosterSetParam(booster, "objective", "binary:logistic");
    XGBoosterSetParam(booster, "eta", "0.1");
    XGBoosterSetParam(booster, "max_depth", "3");
    XGBoosterSetParam(booster, "eval_metric", "logloss");

    // Train the model
    for (int i = 0; i < 10; i++) { // 10 boosting rounds
        XGBoosterUpdateOneIter(booster, i, dtrain);
    }

    // Sample test data (2 samples, 3 features each)
    std::vector<float> test_data = {2.0, 3.0, 4.0,
                                    8.0, 9.0, 10.0};
    int test_rows = 2;
    DMatrixHandle dtest = createDMatrix(test_data, {}, test_rows, cols);

    // Make predictions
    bst_ulong out_len;
    const float* out_result;
    XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result);

    // Output predictions
    std::cout << "Predictions: ";
    for (bst_ulong i = 0; i < out_len; i++) {
        std::cout << out_result[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    XGDMatrixFree(dtrain);
    XGDMatrixFree(dtest);
    XGBoosterFree(booster);

    return 0;
}
