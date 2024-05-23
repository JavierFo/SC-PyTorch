#include <iostream>
#include <vector>

// Define a function to perform 2D convolution with padding and stride
std::vector<std::vector<int> > conv2(const std::vector<std::vector<int> >& input, const std::vector<std::vector<int> >& kernel, int padding, int stride) {
    int inputRows = input.size();
    int inputCols = input[0].size();
    int kernelRows = kernel.size();
    int kernelCols = kernel[0].size();

    // Compute output dimensions considering padding and stride
    int outputRows = (inputRows + 2 * padding - kernelRows) / stride + 1;
    int outputCols = (inputCols + 2 * padding - kernelCols) / stride + 1;

    // Create the output matrix with appropriate dimensions
    std::vector<std::vector<int> > output(outputRows, std::vector<int>(outputCols, 0));

    // Perform 2D convolution with padding and stride
    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            // Compute starting index in input matrix considering stride
            int startRow = i * stride;
            int startCol = j * stride;

            // Compute the convolution sum for the current position
            for (int ki = 0; ki < kernelRows; ++ki) {
                for (int kj = 0; kj < kernelCols; ++kj) {
                    int inputRow = startRow + ki - padding;
                    int inputCol = startCol + kj - padding;

                    // Check for boundary conditions and padding
                    if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols) {
                        output[i][j] += input[inputRow][inputCol] * kernel[ki][kj];
                    }
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

    // Padding and stride parameters
    int padding = 1;
    int stride = 2;

    // Perform convolution with padding and stride
    std::vector<std::vector<int>> result = conv2(input, kernel, padding, stride);

    // Print the result
    for (const auto& row : result) {
        for (int val : row) {
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}*/
