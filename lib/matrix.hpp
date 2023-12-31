#ifndef PROYECTO_FINAL_MATRIX_HPP
#define PROYECTO_FINAL_MATRIX_HPP

#include "../util/libraries.hpp"


template<typename XDT=float>
auto matrix_sparse(const std::size_t &n_rows, const std::size_t &n_columns, float sparsity = 0.1){
    Eigen::SparseMatrix<XDT> sparse_matrix(n_rows,n_columns);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<XDT> dis(0.0, 1.0);

    int non_zero_elements = static_cast<int>(n_rows * n_columns * sparsity); // numero no nulos

    std::vector<Eigen::Triplet<XDT>> tripletList;
    tripletList.reserve(non_zero_elements);

    for (int k = 0; k < non_zero_elements; ++k) {
        int i = gen() % n_rows;
        int j = gen() % n_columns;
        XDT value = static_cast<XDT>(dis(gen));
        tripletList.push_back(Eigen::Triplet<float>(i, j, value));
    }

    sparse_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    //Eigen::Matrix<float,n_rows,n_columns> matrix_zeros;
    //matrix_zeros<XDT,>
    //Eigen::MatrixXd matrix_zeros =
    //matrix_zeros.setZero()
    return sparse_matrix;

}


template <typename XDT=float>
auto matrix_walsh_hadamard_matrix(const std::size_t &d){
    if(d == 1){ // caso base matrix d = 2
        Eigen::MatrixXf cbase{{1,1},{1,-1}};
        return cbase;
    }else{
        Eigen::MatrixXf smaller_matrix = matrix_walsh_hadamard_matrix<XDT>(d / 2);
        Eigen::MatrixXf top = Eigen::MatrixXf::Zero(d, d);
        Eigen::MatrixXf bottom = Eigen::MatrixXf::Zero(d, d);

        top.leftCols(d / 2) = smaller_matrix;
        top.rightCols(d / 2) = smaller_matrix;

        bottom.leftCols(d / 2) = smaller_matrix;
        bottom.rightCols(d / 2) = -smaller_matrix;

        Eigen::MatrixXf result(2 * d, 2 * d);
        result.topRows(d) = top;
        result.bottomRows(d) = bottom;
        return result;
    }
}

int dot_product_mod2(int x, int y) {
    int product = 0;
    while (x && y) {
        product ^= (x % 2) & (y % 2); // XOR de los bits menos significativos
        x >>= 1; // Desplazar x a la derecha
        y >>= 1; // Desplazar y a la derecha
    }
    return product;
}

template<typename XDT>
auto matrix_normalized_walsh_hadamard(const std::size_t &d) {
    Eigen::MatrixXf H(d, d);
    float norm_factor = 1.0 / std::sqrt(d);

    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            H(i, j) = norm_factor * std::pow(-1, dot_product_mod2(i, j));
        }
    }
    return H;
}

auto matrix_diagonal(const std::size_t &d){
    Eigen::MatrixXf D(d, d);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    for (int i = 0; i < d; ++i) {
        D(i, i) = dis(gen) * 2 - 1;
    }
    return D;
}






template<typename XDT=float>
void print_sparse_matrix(const Eigen::SparseMatrix<XDT>& matrix) {
    for (int k = 0; k < matrix.outerSize(); ++k)
        for (Eigen::SparseMatrix<float>::InnerIterator it(matrix, k); it; ++it) {
            std::cout << "Element(" << it.row() << "," << it.col() << ") = \t\t " << it.value() << std::endl;
        }
}

#endif //PROYECTO_FINAL_MATRIX_HPP