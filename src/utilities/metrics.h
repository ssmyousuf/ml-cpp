//
// Created by ml on 11/1/25.
//

#ifndef ML_AI_CPP_UTILS_H
#define ML_AI_CPP_UTILS_H

#include <Eigen/Dense>

namespace utilities {
    class metrics {
    public:

        metrics(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);
        double compute_mse() const;
        double r_square_accuracy();
    private:
        Eigen::VectorXd y_true_;
        Eigen::VectorXd y_pred_;
    };
} // utils

#endif //ML_AI_CPP_UTILS_H