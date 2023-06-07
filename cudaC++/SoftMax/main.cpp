#include <Eigen/Dense>
#include <fstream>
#include <chrono>
#include <iostream>
#include "./cuda_kernel.cuh"
#include <unsupported/Eigen/CXX11/Tensor>
using namespace std;

typedef Eigen::Tensor<float, Eigen::Dynamic, Eigen::RowMajor>
    EigenTensor;

// template <typename T>
// typename TTypes<T>::Flat flat() {
//   return shaped<T, 1>({NumElements()});
// }

int main(){
    streampos size;
    char* memblock;
    ifstream infile("ftmap/1_2.txt", ios::binary | ios::in | ios::ate);
    size = infile.tellg();
    memblock = new char[size];
    infile.seekg(0, ios::beg);
    infile.read(memblock, size);
    infile.close();
    cout << "size is: " << size << " bytes.\n";

    int height = 64, width = 320;
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> input((float*)memblock, height, width);
    // cout<< input<<endl;
    // cout<< input.dimension(0)<<endl;
    // cout<< input.dimension(1)<<endl;
    // float *input_host = input.data();
    // for (int i = 0; i != size / sizeof(float); ++i) {
    //     cout << i << ": " << *(input_host + i) << "\n";
    // }
    Eigen::Tensor<float, 2> temp_t = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 2, Eigen::RowMajor>>(input);
    Eigen::array<int, 2> shuffling({1, 0});
    Eigen::Tensor<float, 2> t = temp_t.shuffle(shuffling);
    // cout << t << endl;
    // float* input_tensor = t.data();
    
    // for (int i = 0; i != size / sizeof(float); ++i) {
    //     cout << i << ": " << *(input_tensor + i) << "\n";
    // }
    // cout<< t.dimension(0)<<endl;
    // cout<< t.dimension(1)<<endl;
    Eigen::array<Eigen::DenseIndex, 2> offsets = {62, 0};
    Eigen::array<Eigen::DenseIndex, 2> extents = {2, 320};
    Eigen::Tensor<float, 2> row_1 = t.slice(offsets, extents).shuffle(shuffling);
    // cout << row_1.size()<<endl;
    // cout<< row_1.dimension(0)<<endl;
    // cout<< row_1.dimension(1)<<endl;
    float *input_host = row_1.data();
    // cout << row_1 <<endl;
    // for (int i = 0; i != 320; ++i) {
    //     cout << i << ": " << *(input_host + i) << "\n";
    // }
    runtest(input_host, 2, width);
    return 0;
}
