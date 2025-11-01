//
// Created by ml on 11/1/25.
//
#include "LinearRegression.h"
#include "metrics.h"
#include "utils.h"
#include <iostream>


int main() {
    // Example usage of LinearRegression class
    algorithm::regression::LinearRegression model(0.0025, 10000);

    // Example: Polynomial regression (y = 2x^2 + 3x + 1)
    VectorXd X(5);
    X << 1, 2, 3, 4, 5;
    VectorXd y = 2 * X.array().square() + 3 * X.array() + 1;

    // Generate polynomial features up to degree 2
    MatrixXd X_poly = utilities::Utils::polynomialFeatures(X, 2);

    // // Sample data: 5 samples, 1 feature
    // Eigen::MatrixXd X(5, 1);
    // X << 1.0,
    //      2.0,
    //      3.0,
    //      4.0,
    //      5.0;

    // Eigen::VectorXd y(5);
    // y << 3.0,
    //      5.0,
    //      7.0,
    //      9.0,
    //      11.0;

    model.fit(X_poly, y);

    std::cout << "Trained Weights: " << model.get_weights() << ", Bias: " << model.get_bias() << std::endl;

    VectorXd predictions = model.predict(X_poly);
    MatrixXd result_compare(y.size(), 2);
    result_compare << y, predictions;
    std::cout << "Actual\tPredicted\n"<< result_compare << std::endl;

    // Evaluate model
    utilities::metrics evaluator(y, predictions);
    double mse = evaluator.compute_mse();
    double acc = evaluator.r_square_accuracy();
    std::cout << "Mean Squared Error: " << mse << std::endl;
    std::cout << "Accuracy (R2 Score): " << acc << std::endl;

    return 0;
}
