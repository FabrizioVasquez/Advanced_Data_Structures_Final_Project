#ifndef MNIST_Image_cpp
#define MNIST_Image_cpp

#include "../include/MNIST_Image.hpp"
#include <cstring>
#include <fstream>

//#include <opencv2/opencv.hpp>

MNIST_Image::MNIST_Image(uint32_t rows, uint32_t cols, int label, char *pixels, int item_id):
_rows(rows),
_cols(cols),
_label(label),
_db_item_id(item_id)
{
    _pixels = new char[_rows * _cols];
    memcpy(_pixels,pixels,rows*cols*sizeof(char));
};

MNIST_Image::MNIST_Image(const MNIST_Image &other):
_rows(other._rows),
_cols(other._cols),
_label(other._label),
_db_item_id(other._db_item_id)
{
    _pixels = new char[_rows*_cols];
    memcpy(_pixels,other._pixels,other._rows*other._cols* sizeof(char));
}

MNIST_Image::~MNIST_Image(){
    delete[] _pixels;
}


//void MNIST_Image::save_as_png(std::string save_dir) {
//    cv::Mat image_tmp(_rows,_cols,CV_8UC1,_pixels);
//    std::string filename = save_dir + "/" + std::to_string(_db_item_id)+ "_" + std::to_string(_label) +".png";
//    cv::imwrite(filename,image_tmp);
//}
//
//void MNIST_Image::save_as_csv(std::string save_filename)
//{
//    std::ofstream outfile;
//    if (_db_item_id == 0)
//        outfile.open(save_filename);
//    else
//        outfile.open(save_filename, std::ios_base::app);
//
//    outfile << _label;
//    for (int p = 0; p < _rows * _cols; p++)
//    {
//        outfile << ',' << std::to_string((unsigned char)_pixels[p]);
//    }
//    outfile << "\n";
//    outfile.close();
//};



#endif /* MNIST_Image_cpp */
