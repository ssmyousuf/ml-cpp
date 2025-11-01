//
// Created by ml on 11/1/25.
//

#ifndef ML_AI_CPP_LINEARREGRESSION_H
#define ML_AI_CPP_LINEARREGRESSION_H

#include <vector>
#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace algorithm {
    namespace regression {
        class LinearRegression {
        public:
            explicit LinearRegression(double learning_rate = 0.01, int n_iterations = 1000, double tolerance = 1e-6);
            LinearRegression() = delete;

            void fit(const MatrixXd& X, const VectorXd& y);
            [[nodiscard]] VectorXd predict(const MatrixXd& X) const ;
            [[nodiscard]] VectorXd get_weights() const { return weights_; }
            [[nodiscard]] double get_bias() const { return bias_; }

        private:
            VectorXd weights_;
            double bias_;
            double learning_rate_;
            int n_iterations_;
            double tolerance_;
            const double TWO_ = 2.0;
            void compute_loss_gradient_descent(const MatrixXd& X, const VectorXd& y);
        };
    } // regression
} // algorithm

#endif //ML_AI_CPP_LINEARREGRESSION_H