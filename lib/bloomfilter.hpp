//
// Created by Fabrizio Vásquez on 18-11-23.
//

#ifndef PROYECTO_FINAL_BLOOMFILTER_HPP
#define PROYECTO_FINAL_BLOOMFILTER_HPP
#include "../util/libraries.hpp"

class BloomFilter {
private:
    std::vector<bool> bit_array;
    int num_hashes;

public:
    BloomFilter(int size, int num_hashes) : bit_array(size, false), num_hashes(num_hashes) {}

    void add(const Eigen::VectorXi &item) {
        for (int i = 0; i < num_hashes; ++i) {
            std::stringstream ss;
            ss << item.transpose().format(Eigen::IOFormat());
            std::string item_str = ss.str() + std::to_string(i);

            std::size_t index = std::hash<std::string>()(item_str) % bit_array.size();
            bit_array[index] = true;
        }
    }


    bool check(const Eigen::VectorXi &item) {
        for (int i = 0; i < num_hashes; ++i) {
            std::stringstream ss;
            ss << item.transpose().format(Eigen::IOFormat());
            std::string item_str = ss.str() + std::to_string(i);

            std::size_t index = std::hash<std::string>()(item_str) % bit_array.size();
            if (!bit_array[index]) {
                return false;
            }
        }
        return true;
    }

};

#endif //PROYECTO_FINAL_BLOOMFILTER_HPP
