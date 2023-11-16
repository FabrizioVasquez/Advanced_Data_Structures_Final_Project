//
// Created by Fabrizio Vásquez on 16-11-23.
//

#ifndef PROYECTO_FINAL_SELECTION_HPP
#define PROYECTO_FINAL_SELECTION_HPP

#include "libraries.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

std::pair<std::pair<MatrixXd, VectorXd>, std::pair<MatrixXd, VectorXd>> train_test_split(const MatrixXd& X, const VectorXd& y, double test_size = 0.2) {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<int> indices(X.rows());
    std::iota(indices.begin(),indices.end(), 0);
    std::shuffle(indices.begin(),indices.end(), g);

    int train_size = static_cast<int>((1.0 - test_size) * X.rows());
    std::vector<int> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<int> test_indices(indices.begin() + train_size, indices.end());

    MatrixXd X_train(train_size, X.cols());
    VectorXd y_train(train_size);
    MatrixXd X_test(X.rows() - train_size, X.cols());
    VectorXd y_test(X.rows() - train_size);

    for (int i = 0; i < train_size; ++i) {
        X_train.row(i) = X.row(train_indices[i]);
        y_train(i) = y(train_indices[i]);
    }
    for (int i = 0; i < X.rows() - train_size; ++i) {
        X_test.row(i) = X.row(test_indices[i]);
        y_test(i) = y(test_indices[i]);
    }

    return {{X_train, y_train}, {X_test, y_test}};
}

#endif //PROYECTO_FINAL_SELECTION_HPP
