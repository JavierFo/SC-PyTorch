#include <iostream>

void conv2(const int input[][7], const int kernel[][3], int output[][7], int inputRows, int inputCols, int kernelRows, int kernelCols, int padding, int stride) {
    const int outputRows = (inputRows - kernelRows + 2 * padding) / stride + 1;
    const int outputCols = (inputCols - kernelCols + 2 * padding) / stride + 1;

    // Perform 2D convolution
    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            // Compute the convolution sum for the current position
            output[i][j] = 0;
            for (int ki = 0; ki < kernelRows; ++ki) {
                for (int kj = 0; kj < kernelCols; ++kj) {
                    int inputRowIndex = i * stride - padding + ki;
                    int inputColIndex = j * stride - padding + kj;
                    if (inputRowIndex >= 0 && inputRowIndex < inputRows && inputColIndex >= 0 && inputColIndex < inputCols) {
                        output[i][j] += input[inputRowIndex][inputColIndex] * kernel[ki][kj];
                    }
                }
            }
        }
    }
}

/*int main() {
    // Example input matrix
    int input[7][7] = {
        {1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14},
        {15, 16, 17, 18, 19, 20, 21},
        {22, 23, 24, 25, 26, 27, 28},
        {29, 30, 31, 32, 33, 34, 35},
        {36, 37, 38, 39, 40, 41, 42},
        {43, 44, 45, 46, 47, 48, 49}
    };

    // Example kernel matrix
    int kernel[3][3] = {
        {1, 0, 11},
        {2, 0, 12},
        {3, 0, 13}
    };

    // Output matrix
    int output[7][7];

    // Define padding and stride
    int padding = 1;
    int stride = 1;

    // Perform convolution
    conv2(input, kernel, output, 7, 7, 3, 3, padding, stride);

    // Print the result
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            std::cout << output[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}*/
