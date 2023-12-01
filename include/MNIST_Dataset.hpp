#include "MNIST_Image.hpp"
#include <vector>

class MNIST_Dataset{
private:
    std::vector<MNIST_Image> _images;
    const char*_image_filename;
    const char*_label_filename;
    int _image_magic;
    int _label_magic;
public:
    MNIST_Dataset(const char *image_filename,
                  const char *label_filename,
                  int image_magic,
                  int label_magic);

    void read_mnist_db(const int max_items);
    std::size_t get_images_length();
    int get_label_from_index(int index);

    void save_dataset_as_png(std::string save_dir);
    void save_dataset_as_csv(std::string save_dir);
    Eigen::MatrixXf to_matrix();
    static Eigen::MatrixXf get_X(Eigen::MatrixXf &mat);
    static Eigen::VectorXf get_Y(Eigen::MatrixXf &mat);
};