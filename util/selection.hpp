//
// Created by Fabrizio Vásquez on 16-11-23.
//

#ifndef PROYECTO_FINAL_SELECTION_HPP
#define PROYECTO_FINAL_SELECTION_HPP

#include "libraries.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

std::pair<std::pair<Eigen::MatrixXf, Eigen::VectorXf>, std::pair< Eigen::MatrixXf, Eigen::VectorXf>> train_test_split(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, double test_size = 0.2) {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<int> indices(X.rows());
    std::iota(indices.begin(),indices.end(), 0);
    std::shuffle(indices.begin(),indices.end(), g);

    int train_size = static_cast<int>((1.0 - test_size) * X.rows());
    std::vector<int> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<int> test_indices(indices.begin() + train_size, indices.end());

    Eigen::MatrixXf X_train(train_size, X.cols());
    Eigen::VectorXf y_train(train_size);
    Eigen::MatrixXf X_test(X.rows() - train_size, X.cols());
    Eigen::VectorXf y_test(X.rows() - train_size);

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

std::pair<double, double> calcular_DR_FAR(const std::vector<int>& y_real, const std::vector<int>& y_pred) {
    int TP = 0, FP = 0, TN = 0, FN = 0;

    for (size_t i = 0; i < y_real.size(); ++i) {
        if (y_real[i] == 1 && y_pred[i] == 1) {
            TP++;
        } else if (y_real[i] == 0 && y_pred[i] == 1) {
            FP++;
        } else if (y_real[i] == 0 && y_pred[i] == 0) {
            TN++;
        } else if (y_real[i] == 1 && y_pred[i] == 0) {
            FN++;
        }
    }

    //std::cout<<TP;

    double DR = static_cast<double>(TP + TN) / (TP + TN + FP + FN);
    double FAR = static_cast<double>(FP + FN) / (TP + TN + FP + FN);

    return {DR, FAR};
}

std::vector<int> eigen_vector_to_std_vector(const Eigen::VectorXf& eigen_vector) {
    std::vector<int> vec;
    vec.reserve(eigen_vector.size());
    for (int i = 0; i < eigen_vector.size(); ++i) {
        vec.push_back(static_cast<int>(eigen_vector[i]));
    }
    return vec;
}


#endif //PROYECTO_FINAL_SELECTION_HPP
