#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <algorithm>
#include "util/read.hpp"
#include "util/selection.hpp"
#include "lib/matrix.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;


int main() {
    MatrixXd matrixXd{{1,0},{0,1}};
    MatrixXd xy{{4},{5}};
    //std::cout<<(xy.transpose() * matrixXd);

    MatrixXd data = read_csv("../data/speech_data.csv");
    // VectorXd last = data.col(data.cols()-1);
    //std::cout<<last.rows()<<"\n";
    //std::cout<<last.cols();
    Eigen::MatrixXd X =  data.leftCols(data.cols()-1);
    Eigen::VectorXd y = data.rightCols(1);
    //std::cout<<y;
    auto [trainSet, testSet] = train_test_split(X, y);

    Eigen::MatrixXd X_train = trainSet.first;
    Eigen::VectorXd y_train = trainSet.second;
    Eigen::MatrixXd X_test = testSet.first;
    Eigen::VectorXd y_test = testSet.second;

    long n = X_train.rows(); // rows or data.rows
    long d = X_train.cols(); // cols or dimension
    long k = 10;             // hash-code
    long m = 100;            // size of blooom filter


    // PRIMER PASO CONSTRUIR LA MATRIZ
    // n: filas
    // d: columnas
    // b: reduccion filas

    Eigen::SparseMatrix P = matrix_sparse<float>(10, d, 0.1);
    //Eigen::MatrixXf    H = matrix_walsh_hadamard_matrix<float>(400);
    Eigen::MatrixXf      H = matrix_normalized_walsh_hadamard<float>(d);
    Eigen::MatrixXf      D = matrix_diagonal(d);
    Eigen::MatrixXf    PHD = P*H*D;

    Eigen::MatrixXf Y = X_train.cast<float>() * PHD.transpose();


    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dMat;
    // Sparse to dense:
    dMat = MatrixXf(P);
    // Dense to sparse:
    P = dMat.sparseView();

    std::cout<<Y.rows();


    return 0;
}
