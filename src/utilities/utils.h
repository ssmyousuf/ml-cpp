//
// Created by ml on 11/1/25.
//
#pragma once
#include <Eigen/Dense>
namespace utilities {

    class Utils {
    public:
        static Eigen::MatrixXd polynomialFeatures(const Eigen::VectorXd& x, int degree);

        static void train_test_split(
            const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
            double test_ratio,
            Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train,
            Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test);
    };
} // namespace utilities
