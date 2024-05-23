#include <iostream>
#include <vector>

std::vector<std::vector<int> > conv2(const std::vector<std::vector<int>>& input, const std::vector<std::vector<int>>& kernel) {
    int inputRows = input.size();
    int inputCols = input[0].size();
    int kernelRows = kernel.size();
    int kernelCols = kernel[0].size();

    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;

    // Create the output matrix with appropriate dimensions
    std::vector<std::vector<int>> output(outputRows, std::vector<int>(outputCols, 0));

    // Perform 2D convolution
    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            // Compute the convolution sum for the current position
            for (int ki = 0; ki < kernelRows; ++ki) {
                for (int kj = 0; kj < kernelCols; ++kj) {
                    output[i][j] += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
        }
    }
    return output;
}

/*int main() {
    // Example input matrix
    std::vector<std::vector<int>> input = {
        {1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14},
        {15, 16, 17, 18, 19, 20, 21},
        {22, 23, 24, 25, 26, 27, 28},
        {29, 30, 31, 32, 33, 34, 35},
        {36, 37, 38, 39, 40, 41, 42},
        {43, 44, 45, 46, 47, 48, 49}
    };


    // Example kernel matrix
    std::vector<std::vector<int>> kernel = {
        {1, 0, 11},
        {2, 0, 12},
        {3, 0, 13}
    };

    // Perform convolution
    std::vector<std::vector<int>> result = conv2(input, kernel);

    // Print the result
    for (const auto& row : result) {
        for (int val : row) {
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}
*/