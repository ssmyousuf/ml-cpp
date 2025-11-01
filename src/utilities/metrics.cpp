//
// Created by ml on 11/1/25.
//

#include "metrics.h"
#include <iostream>


namespace utilities {
    metrics::metrics (const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred)
        : y_true_(y_true), y_pred_(y_pred) {}

    double metrics::compute_mse() const {
        const Eigen::VectorXd error = y_pred_ - y_true_;
        return error.squaredNorm() / y_true_.size();
    }

    double metrics::r_square_accuracy() {
        if (y_true_.size() != y_pred_.size()) {
            std::cerr << "Size mismatch between true labels and predictions." << std::endl;
            return 0.0;
        }
        // R2 like accuracy
        const double mse = compute_mse();
        // Variance of y_true
        const double var = (y_true_.array() - y_true_.mean()).square().sum() / y_true_.size();
        return 1.0 - mse / var; //R2 Score = 1 (MSE /vairance)
    }

} // utils