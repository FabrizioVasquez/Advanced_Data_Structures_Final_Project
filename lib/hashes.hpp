//
// Created by Fabrizio Vásquez on 16-11-23.
//

#ifndef PROYECTO_FINAL_HASHES_HPP
#define PROYECTO_FINAL_HASHES_HPP
#include "../util/libraries.hpp"

auto hashes(
        const Eigen::MatrixXf& X,
        const Eigen::SparseMatrix<float>& P,
        const Eigen::MatrixXf& H,
        const Eigen::MatrixXf& D,
        const Eigen::MatrixXf& M,
        const Eigen::MatrixXf& a){
    Eigen::MatrixXf Y = X * (P * H * D).transpose();

    Eigen::MatrixXi hash_codes(Y.rows(), M.rows());

    for (int i = 0; i < Y.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            hash_codes(i, j) = std::floor(Y.row(i).dot(M.col(j)) + a(j));
        }
    }

    Eigen::MatrixXi binary_codes(Y.rows(), M.rows());

    for (int i = 0; i < hash_codes.rows(); ++i) {
        for (int j = 0; j < hash_codes.cols(); ++j) {
            if (hash_codes(i, j) >= 0) {
                binary_codes(i, j) = 1;
            } else {
                binary_codes(i, j) = 0;
            }
        }
    }
    // Satisface R^n*k -> n * M.rows() familia de hashes
    return binary_codes;
}

#endif //PROYECTO_FINAL_HASHES_HPP
