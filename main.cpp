#include <iostream>
#include <Eigen/Dense>
#include "util/read.hpp"
#include "util/selection.hpp"
#include "lib/matrix.hpp"
#include "lib/hashes.hpp"
#include "lib/bloomfilter.hpp"
#include "SimpleIni.h"
#include "include/MNIST_Dataset.hpp"
#include <nlohmann/json.hpp>

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049


std::vector<int> paper_algorithm(const Eigen::MatrixXf& X_train,
                                 const Eigen::MatrixXf& X_test)
{

    long n = X_train.rows(); // número de datos de entrenamiento
    long d = X_train.cols(); // dimensión original
    long k = 50;             // cantidad de hash codes
    long m = 100;           // tamaño del filtro de Bloom
    long b = 50;             // dimensión reducida

    Eigen::SparseMatrix<float> P1 = matrix_sparse<float>(b, d, 0.1);
    MatrixXf H1 = matrix_normalized_walsh_hadamard<float>(d);
    MatrixXf D1 = matrix_diagonal(d);

    Eigen::SparseMatrix<float> P2 = matrix_sparse<float>(b, d, 0.1);
    MatrixXf H2 = matrix_normalized_walsh_hadamard<float>(d);
    MatrixXf D2 = matrix_diagonal(d);

    MatrixXf M1 = Eigen::MatrixXf::Random(k, b);
    VectorXf a1 = Eigen::VectorXf::Random(k);
    MatrixXf M2 = Eigen::MatrixXf::Random(k, b);
    VectorXf a2 = Eigen::VectorXf::Random(k);

    MatrixXi binary_code_train1 = hashes(X_train, P1, H1, D1, M1, a1);
    MatrixXi binary_code_train2 = hashes(X_train, P2, H2, D2, M2, a2);

    BloomFilter bloom_filter1(m, k);
    BloomFilter bloom_filter2(m, k);

    for (int i = 0; i < X_train.rows(); ++i) {
        VectorXi code1 = binary_code_train1.row(i);
        VectorXi code2 = binary_code_train2.row(i);
        bloom_filter1.add(code1);
        bloom_filter2.add(code2);
    }

    MatrixXi binary_code_test1 = hashes(X_test, P1, H1, D1, M1, a1);
    MatrixXi binary_code_test2 = hashes(X_test, P2, H2, D2, M2, a2);

    std::vector<int> y_pred;
    for (int i = 0; i < X_test.rows(); ++i) {
        VectorXi codeTest1 = binary_code_test1.row(i);
        VectorXi codeTest2 = binary_code_test2.row(i);

        if (bloom_filter1.check(codeTest1) && bloom_filter2.check(codeTest2)) {
            y_pred.push_back(0);
        } else {
            y_pred.push_back(1);
        }
    }

    return y_pred;
}


void dLSHBF_MNIST(const Eigen::MatrixXf& X_train, const Eigen::VectorXf& y_train,
                  const Eigen::MatrixXf& X_test, const Eigen::VectorXf& y_test) {

    std::vector<int> y_pred = paper_algorithm(X_train, X_test);

    std::cout << "Resultado de la MNIST DATASET: ";
    int count2 = 0;
    //std::cout << "REAL    vs PREDECIDO" << std::endl;

    for (int i = 0; i < y_pred.size(); i++) {
        //std::cout << y_test[i] << " " << y_pred[i] << std::endl;
        if (y_test[i] == 1) {
            count2 += 1;
        }
    }

    int count = 0;
    for (auto &i : y_pred) {
        if (i == 1) {
            count += 1;
        }
    }

    // Crear un nuevo vector para almacenar los resultados transformados
    Eigen::VectorXi y_test_transformed(y_test.size());

    // Transformar 'y_test' para que contenga solo 0 y 1
    for (int i = 0; i < y_test.size(); ++i) {
        if (y_test(i) != 1) {
            y_test_transformed(i) = 0;  // Cambiar todos los valores diferentes de 1 a 0
        } else {
            y_test_transformed(i) = 1;  // Mantener los valores que son 1
        }
    }

// Convertir 'Eigen::VectorXi' a 'std::vector<int>'

    std::vector<int> y_test_std;
    y_test_std.reserve(y_test_transformed.size());
    for (int i = 0; i < y_test_transformed.size(); ++i) {
        y_test_std.push_back(y_test_transformed[i]);
    }
    std::pair<double, double> result = calcular_DR_FAR(y_test_std, y_pred);
    std::cout << "DR: " << result.first << ", FAR: " << result.second << std::endl;
}

auto dLSHBF_SPEECH() {
    MatrixXf data = read_csv("../data/speech_data.csv");
    Eigen::MatrixXf X = data.leftCols(data.cols() - 1);
    Eigen::VectorXf y = data.rightCols(1);
    auto [trainSet, testSet] = train_test_split(X, y);
    Eigen::MatrixXf X_train = trainSet.first;
    Eigen::VectorXf y_train = trainSet.second;
    Eigen::MatrixXf X_test = testSet.first;
    Eigen::VectorXf y_test = testSet.second;

    long n = X_train.rows(); // número de datos de entrenamiento
    long d = X_train.cols(); // dimensión original
    long k = 100;             // cantidad de hash codes
    long m = 1000;           // tamaño del filtro de Bloom
    long b = 100;             // dimensión reducida

    Eigen::SparseMatrix<float> P1 = matrix_sparse<float>(b, d, 0.1);
    MatrixXf H1 = matrix_normalized_walsh_hadamard<float>(d);
    MatrixXf D1 = matrix_diagonal(d);

    Eigen::SparseMatrix<float> P2 = matrix_sparse<float>(b, d, 0.1);
    MatrixXf H2 = matrix_normalized_walsh_hadamard<float>(d);
    MatrixXf D2 = matrix_diagonal(d);

    MatrixXf M1 = Eigen::MatrixXf::Random(k, b);
    VectorXf a1 = Eigen::VectorXf::Random(k);
    MatrixXf M2 = Eigen::MatrixXf::Random(k, b);
    VectorXf a2 = Eigen::VectorXf::Random(k);

    MatrixXi binary_code_train1 = hashes(X_train, P1, H1, D1, M1, a1);
    MatrixXi binary_code_train2 = hashes(X_train, P2, H2, D2, M2, a2);

    BloomFilter bloom_filter1(m, k);
    BloomFilter bloom_filter2(m, k);

    for (int i = 0; i < X_train.rows(); ++i) {
        VectorXi code1 = binary_code_train1.row(i);
        VectorXi code2 = binary_code_train2.row(i);
        bloom_filter1.add(code1);
        bloom_filter2.add(code2);
    }

    MatrixXi binary_code_test1 = hashes(X_test, P1, H1, D1, M1, a1);
    MatrixXi binary_code_test2 = hashes(X_test, P2, H2, D2, M2, a2);

    std::vector<int> y_pred;
    for (int i = 0; i < X_test.rows(); ++i) {
        VectorXi codeTest1 = binary_code_test1.row(i);
        VectorXi codeTest2 = binary_code_test2.row(i);

        if (bloom_filter1.check(codeTest1) && bloom_filter2.check(codeTest2)) {
            y_pred.push_back(0);
        } else {
            y_pred.push_back(1);
        }
    }

    std::cout << "Resultado de la SPEECH DATASET: ";
    int count2 = 0;
    //std::cout << "REAL    vs PREDECIDO" << std::endl;

    for (int i = 0; i < y_pred.size(); i++) {
        //std::cout << y_test[i] << " " << y_pred[i] << std::endl;
        if (y_test[i] == 1) {
            count2 += 1;
        }
    }

    int count = 0;
    for (auto &i : y_pred) {
        if (i == 1) {
            count += 1;
        }
    }
    std::vector<int> y_test_std     = eigen_vector_to_std_vector(y_test);
    std::pair<double, double> result = calcular_DR_FAR(y_test_std, y_pred);
    std::cout << "DR: " << result.first << ", FAR: " << result.second << std::endl;
}

int main(int argc, char **argv) {
    CSimpleIni ini;
    ini.SetUnicode();

    std::string basepath = "/home/miunmn/Documents/utec/eda/proyecto_final_EDA";

    SI_Error rc = ini.LoadFile("/home/miunmn/Documents/utec/eda/proyecto_final_EDA/config.ini");
    if(rc < 0){
        std::cout << "Error loading file" << std::endl;
        return EXIT_FAILURE;
    }
    SI_ASSERT(rc == SI_OK);

    // TRAIN PATHS

    std::string base_dir       = ini.GetValue("MNIST","BASE_DIR","MNIST");
    std::string save_dir_train       = base_dir + "/train";
    std::string img_filename_train   = ini.GetValue("MNIST","TRAIN_IMAGE_FILE","train-images-idx3-ubyte");
    std::string img_path_train       = base_dir + "/" + img_filename_train;
    std::string label_filename_train = ini.GetValue("MNIST","TRAIN_LABEL_FILE","train-labels-idx1-ubyte");
    std::string label_path_train     = base_dir + "/" + label_filename_train;

    // TEST PATHS

    std::string save_dir_test       = base_dir + "/test";
    std::string img_filename_test   = ini.GetValue("MNIST", "TEST_IMAGE_FILE", "t10k-images-idx3-ubyte");
    std::string img_path_test       = base_dir + "/" + img_filename_test;
    std::string label_filename_test = ini.GetValue("MNIST", "TEST_LABEL_FILE", "t10k-labels-idx1-ubyte");
    std::string label_path_test     = base_dir + "/" + label_filename_test;



    int num_generations = ini.GetLongValue("MNIST","GENERATIONS",5);
    int max_items       = ini.GetLongValue("MNIST","MAX_ITEMS",15);
    bool save_img       = ini.GetBoolValue("MNIST","SAVE_IMG",false);
    float alpha         = ini.GetDoubleValue("MNIST","ALPHA",0.1);
    int hidden_layer_size = ini.GetLongValue("MNIST","HIDDEN_LAYER_SIZE",10);
    int k = ini.GetLongValue("MNIST","HASH_CODES",10);
    int m = ini.GetLongValue("MNIST","FILTRO_BLOOM",1000);
    int b = ini.GetLongValue("MNIST","REDUCE_DIMENSION",10);

    std::cout << PRINT_VAR(num_generations) <<" "
              << PRINT_VAR(max_items)       <<" "
              << PRINT_VAR(save_img)        <<" "
              << PRINT_VAR(alpha)           <<" "
              << PRINT_VAR(hidden_layer_size) << std::endl;

    MNIST_Dataset train_dataset(img_path_train.c_str(),label_path_train.c_str(),TRAIN_IMAGE_MAGIC,TRAIN_LABEL_MAGIC);
    //std::cout<<img_path;
    train_dataset.read_mnist_db(60000);

    save_dir_train = base_dir + "/test";
    img_filename_train = ini.GetValue("MNIST", "TEST_IMAGE_FILE", "t10k-images-idx3-ubyte");
    img_path_train = base_dir + "/" + img_filename_train;
    label_filename_train = ini.GetValue("MNIST", "TEST_LABEL_FILE", "t10k-labels-idx1-ubyte");
    label_path_train = base_dir + "/" + label_filename_train;
    std::cout<<img_path_train;

    MNIST_Dataset test_dataset(img_path_train.c_str(), label_path_train.c_str(), TEST_IMAGE_MAGIC, TEST_LABEL_MAGIC);
    test_dataset.read_mnist_db(10000);

    Eigen::MatrixXf train_mat = train_dataset.to_matrix();
    Eigen::MatrixXf test_mat = test_dataset.to_matrix();

    std::vector<int> normal_indices;
    std::vector<int> abnormal_indices;
    for (int i = 0; i < train_mat.rows(); ++i) {
        if (train_mat(i, 0) != 1) {
            normal_indices.push_back(i);
        } else {
            abnormal_indices.push_back(i);
        }
    }

    Eigen::MatrixXf normal_data_train(normal_indices.size(), train_mat.cols());
    Eigen::MatrixXf abnormal_data_train(abnormal_indices.size(), train_mat.cols());
    for (size_t i = 0; i < normal_indices.size(); ++i) {
        normal_data_train.row(i) = train_mat.row(normal_indices[i]);
    }
    for (size_t i = 0; i < abnormal_indices.size(); ++i) {
        abnormal_data_train.row(i) = train_mat.row(abnormal_indices[i]);
    }

    int train_size = static_cast<int>(0.8 * normal_data_train.rows());
    Eigen::MatrixXf X_train = normal_data_train.topRows(train_size).rightCols(normal_data_train.cols() - 1);
    Eigen::VectorXf y_train = normal_data_train.topRows(train_size).leftCols(1);

    Eigen::MatrixXf X_test_normal = normal_data_train.bottomRows(normal_data_train.rows() - train_size).rightCols(normal_data_train.cols() - 1);
    Eigen::VectorXf y_test_normal = normal_data_train.bottomRows(normal_data_train.rows() - train_size).leftCols(1);

    Eigen::MatrixXf X_test_abnormal = abnormal_data_train.rightCols(abnormal_data_train.cols() - 1);
    Eigen::VectorXf y_test_abnormal = abnormal_data_train.leftCols(1);

    X_train /= 255.0;
    X_test_normal /= 255.0;
    X_test_abnormal /= 255.0;

    Eigen::MatrixXf X_test(X_test_normal.rows() + X_test_abnormal.rows(), X_test_normal.cols());
    Eigen::VectorXf y_test(y_test_normal.size() + y_test_abnormal.size());

    X_test << X_test_normal, X_test_abnormal;
    y_test << y_test_normal, y_test_abnormal;

    std::cout<<std::endl;
    std::cout<<"TRAIN Y TEST Y"<<std::endl;
    std::cout<<"Filas Y train: "<<y_train.rows()<<std::endl;
    std::cout<<"Filas Y test: "<<y_test.rows()<<" ";
    std::cout<<y_test.cols();
    std::cout<<std::endl;
    std::cout<<"TRAIN Y TEST X"<<std::endl;
    std::cout<<"Filas X train: "<<X_train.rows()<<std::endl;
    std::cout<<"Filas X test: "<<X_test.rows()<<" ";
    std::cout<<std::endl;
    //std::cout<<X_test;
    //std::cout<<y_test;

    std::vector<int> normal_indices_;
    std::vector<int> abnormal_indices_;
    for (int i = 0; i < train_mat.rows(); ++i) {
        if (train_mat(i, 0) != 1) {
            normal_indices_.push_back(i);
        } else {
            abnormal_indices_.push_back(i);
        }
    }

//    std::cout << "Total normal data: " << normal_indices.size() << std::endl;
//    std::cout << "Total anormal data: " << abnormal_indices.size() << std::endl;
//    std::cout<< "\n X_test: " << X_test;

//    X_test leer de csv
    std::ifstream file("/home/miunmn/Documents/utec/eda/api-final/points.json");
    nlohmann::json j;
    file >> j;

    // Acceder a los datos analizados
    std::vector<double> points = j["points"].get<std::vector<double>>();

    // Imprimir los puntos
    Eigen::MatrixXf user_input(1,784);
    user_input.setZero();
    std::cout << "Puntos: ";
    for (int i = 0; i < points.size(); i++) {
        user_input(0, i) = points[i];
        std::cout << points[i] << " ";
    }



      auto result = paper_algorithm(X_train, user_input);
      std::string store_string = std::to_string(result[0]);
    std::cout<<"storing: "<<store_string<<'\n';
    const char* fileName = "/home/miunmn/Documents/utec/eda/api-final/result.txt";

    std::ofstream outputFile(fileName);
    if (outputFile.is_open()) {
        // Write the string to the file
        outputFile << store_string << std::endl;

        // Close the file
        outputFile.close();

        std::cout << "String has been successfully stored in " << fileName << std::endl;
    } else {
        std::cerr << "Error opening the file: " << fileName << std::endl;
    }
//    dLSHBF_MNIST(X_train, y_train, X_test, y_test);
//    dLSHBF_SPEECH();

    return EXIT_SUCCESS;
}
