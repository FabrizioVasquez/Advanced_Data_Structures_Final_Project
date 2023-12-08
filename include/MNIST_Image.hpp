#ifndef MNIST_Image_hpp
#define MNIST_Image_hpp

#pragma once
#include <cstdint>

class MNIST_Image{
public:
    uint32_t _rows;
    uint32_t _cols;
    int _label;
    char *_pixels;
    int _db_item_id;
    MNIST_Image(uint32_t rows, uint32_t cols, int label, char *pixels, int item_id);
    MNIST_Image(const MNIST_Image &other);
    ~MNIST_Image();

//    void save_as_png(std::string save_dir);
//    void save_as_csv(std::string save_filename);
};


#endif /* MNIST_Image_hpp */
