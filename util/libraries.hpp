//
// Created by Fabrizio Vásquez on 16-11-23.
//

#ifndef PROYECTO_FINAL_LIBRARIES_HPP
#define PROYECTO_FINAL_LIBRARIES_HPP
#pragma once
#define PRINT_VAR(x) #x << "=" << x
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <random>
#include <algorithm>
#include <cstdint>

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::VectorXi;

#endif //PROYECTO_FINAL_LIBRARIES_HPP
