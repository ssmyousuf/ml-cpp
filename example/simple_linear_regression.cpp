//
// Created by ml on 11/1/25.
//

#include <vector>
#include <iostream>
//Predict
// double Predict(const std::vector<double>& features, const std::vector<double>& weights, double bias) {
//     double result = bias;
//     for (size_t i = 0; i < features.size(); ++i) {
//         result += features[i] * weights[i];
//     }
//     return result;
// }

// double Predict(const std::vector<std::vector<double>>& feature_matrix, const std::vector<double>& weights, double bias) {
//     std::vector<double> predictions;
//     for (const auto& features : feature_matrix) {
//         predictions.push_back(Predict(features, weights, bias));
//     }
//     return predictions;
// }

double Predict(double feature, double weight, double bias) {
    return bias + feature * weight;
}

// Compute Loss (Mean Squared Error)
double ComputeLoss(const std::vector<double>& feature, const std::vector<double>& targets, double weight, double bias) {

    double total_cost = 0.0;
    size_t m = feature.size();
    for (size_t i = 0; i < m; ++i) {
        double prediction = Predict(feature[i], weight, bias);
        double target = targets[i];
        double error = prediction - target;
        total_cost += error * error;
    }
    return total_cost / (2 * m);
}


//Perfrom Gradient Descent
void GradientDescent(const std::vector<double>& feature, const std::vector<double>& targets, double& weight, double& bias, double learning_rate) {
    size_t m = feature.size();
    double dw = 0.0;
    double db = 0.0;
    for (size_t i = 0; i < m; ++i) {
            double prediction = Predict(feature[i], weight, bias);
            double error = prediction - targets[i];
            dw += error * feature[i];
            db += error;
        }
        dw /= m;
        db /= m;

        weight -= learning_rate * dw;
        bias -= learning_rate * db;
}

int main(int argc, const char** argv) {
    // Example data set y = 2x+ 1
    std::vector<double> feature = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> targets = {3.0, 5.0, 7.0, 9.0, 11.0};

    double weight = 0.0; // Initial weight
    double bias = 0.0; // Initial bias
    double learning_rate = 0.01; // Learning rate
    int iterations = 1000; // Number of iterations
    for (int i = 0; i < iterations; ++i) {
        GradientDescent(feature, targets, weight, bias, learning_rate);
        if (i % 100 == 0) {
            double loss = ComputeLoss(feature, targets, weight, bias);
            std::cout << "Iteration " << i << ": Loss = " << loss << ", Weight = " << weight << ", Bias = " << bias << std::endl;
        }
    }
    std::cout << "Trained Weight: " << weight << ", Trained Bias: " << bias << std::endl;
    // Test prediction
    double test_feature = 6.0;
    double prediction = Predict(test_feature, weight, bias);
    std::cout << "Prediction for feature " << test_feature << ": " << prediction << std::endl;
    return 0;
}


