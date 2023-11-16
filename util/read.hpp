//
// Created by Fabrizio Vásquez on 13-11-23.
//
#include "libraries.hpp"
#include <fstream>

MatrixXd read_csv(const std::string& _path){
    const std::string& filename{_path};
    std::ifstream file(filename.c_str());
    if(!file.is_open()){
        std::cout<<"No se encuentra el archivo "<<filename<<".\n";
    }
    std::vector<std::vector<double>> data;
    std::string line{};
    getline(file,line);
    while(getline(file,line)){

        std::stringstream csvStream(line);
        std::string cell;
        std::vector<double> fila;
        while(getline(csvStream,cell,',')){
            fila.push_back(std::stod(cell));
        }

        data.push_back(fila);
    }
    size_t rows = data.size(), columns = data.front().size();
    MatrixXd matrix(rows, columns);
    for(int i = 0; i<rows; i++){
        for(int j = 0; j<columns; j++){
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;

}
