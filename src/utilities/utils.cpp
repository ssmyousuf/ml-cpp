//
// Created by ml on 11/1/25.
//

#include "utils.h"
#include <random>

namespace utilities {
    Eigen::MatrixXd Utils::polynomialFeatures(const Eigen::VectorXd& x, int degree) {
        int m = x.size();
        Eigen::MatrixXd X_poly(m, degree);
        for (int i = 0; i < m; ++i) {
            for (int d = 0; d < degree; ++d) {
                X_poly(i, d) = pow(x(i), d + 1);
            }
        }
        return X_poly;
    }

    // Split function
    void Utils::train_test_split(
        const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
        double test_ratio,
        Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train,
        Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test) {
        int n_samples = X.rows();
        int n_test = static_cast<int>(n_samples * test_ratio);
        int n_train = n_samples - n_test;

        // Random shuffle indices
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        // Create train/test
        X_train = Eigen::MatrixXd(n_train, X.cols());
        y_train = Eigen::VectorXd(n_train);
        X_test = Eigen::MatrixXd(n_test, X.cols());
        y_test = Eigen::VectorXd(n_test);

        for (int i = 0; i < n_train; ++i) {
            X_train.row(i) = X.row(indices[i]);
            y_train(i) = y(indices[i]);
        }
        for (int i = 0; i < n_test; ++i) {
            X_test.row(i) = X.row(indices[n_train + i]);
            y_test(i) = y(indices[n_train + i]);
        }
    }
}