#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
    MatrixXd matrixXd{{1,0},{0,1}};
    MatrixXd xy{{4},{5}};

    //std::cout<<matrixXd.transpose();
    //std::cout<<xy;

    //VectorXd vec = static_cast<VectorXd>(xy.transpose() * matrixXd);

    std::cout<<(xy.transpose() * matrixXd);
    //std::cout<<(xy2 * matrixXd);
    return 0;
}
