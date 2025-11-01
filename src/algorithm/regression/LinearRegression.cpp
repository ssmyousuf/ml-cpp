//
// Created by ml on 11/1/25.
//

#include "LinearRegression.h"
#include  <iostream>

using namespace std;

namespace algorithm::regression {
    LinearRegression::LinearRegression(const double learning_rate, const int n_iterations, const double tolerance)
        : learning_rate_(learning_rate), n_iterations_(n_iterations), bias_(0.0), tolerance_(tolerance) {
    }

    VectorXd LinearRegression::predict(const MatrixXd &X) const {
        // return (X * weights_).array() + bias_;
        return (X * weights_) + VectorXd::Ones(X.rows()) * bias_;
    }

    void LinearRegression::fit(const MatrixXd &X, const VectorXd &y) {

        int n_features = X.cols(); // Number of features
        weights_ = VectorXd::Zero(n_features); // Initialize weights to zero
        compute_loss_gradient_descent(X, y);
        std::cout << "Training completed. Weights and bias updated." << std::endl;
        std::cout << "Weights: " << weights_ << ", Bias: " << bias_ << std::endl;
    }

    void LinearRegression::compute_loss_gradient_descent(const MatrixXd &X, const VectorXd &y) {
        // Implementation of loss computation and gradient descent
        int n_samples = X.rows(); // Number of samples
        double prev_loss = std::numeric_limits<double>::max();
        for (int i = 0; i < n_iterations_; ++i) {
            // Gradient descent logic here
            VectorXd target_prediction = predict(X);
            VectorXd error = target_prediction - y;

            double loss = (error.array().square().sum()) / n_samples;

            // Stop automatically when cost improvement is too small
            if (std::abs(prev_loss - loss) < tolerance_) {
                std::cout << "Early stopping at iteration " << i
                        << " | Loss: " << loss << std::endl;
                break;
            }
            prev_loss = loss;

            // Compute gradients
            VectorXd dw = 2.0 / n_samples * X.transpose() * error;
            const double db = 2.0 / n_samples * error.sum();
            // Update weights and bias
            weights_ -= learning_rate_ * dw;
            bias_ -= learning_rate_ * db;

            if (i % 100 == 0) {
                loss = (error.array().square().sum()) / n_samples;
                // Optionally print loss
                std::cout << "Iteration " << i << ", Loss: " << loss << ", Weights: " << weights_.transpose() <<
                        ", Bias: " << bias_ << std::endl;
            }
        }
    }
} // regression namespace algorithm
