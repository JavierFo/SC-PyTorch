#include <iostream>
#include <vector>
#include "StochasticTensor.h"

std::vector<std::vector<double> > SC_conv2(const StochasticTensor input, const StochasticTensor kernel) {
    StochasticTensor SCtensorInput = input;
    StochasticTensor SCtensorKernel = kernel;

    StochasticTensor::SizeTuple inputSizes = SCtensorInput.getSize();
    StochasticTensor::SizeTuple kernelSizes = SCtensorKernel.getSize();
    
    const int inputRows = std::get<0>(inputSizes);
    const int inputCols = std::get<1>(inputSizes);
    const int kernelRows = std::get<0>(kernelSizes);
    const int kernelCols = std::get<1>(kernelSizes);

    const int outputRows = inputRows - kernelRows + 1;
    const int outputCols = inputCols - kernelCols + 1;

    // Create the output matrix with appropriate dimensions
    std::vector<std::vector<double>> output(outputRows, std::vector<double>(outputCols, 0));

    // Perform 2D convolution
    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            // Compute the convolution sum for the current position
            for (int ki = 0; ki < kernelRows; ++ki) {
                for (int kj = 0; kj < kernelCols; ++kj) {
                    //output[i][j] += input[i + ki][j + kj] * kernel[ki][kj];
                    std::vector<int> input_kernel_SCMultiplication = bitstreamOperation(input.getVectorAt(i + ki, j + kj), kernel.getVectorAt(ki,kj), XNOR);
                    output[i][j] += calculatePx(input_kernel_SCMultiplication, BIPOLAR);
                    //output[i][j] += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
        }
    }
    return output;
}

void prettyPrint(const std::vector<std::vector<double>>& matrix) {
    std::cout << "{\n";
    for (const auto& row : matrix) {
        std::cout << "  { ";
        for (size_t i = 0; i < row.size(); ++i) {
            std::cout << row[i];
            if (i < row.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " }";
        if (&row != &matrix.back()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "}\n";
}

int main() {
    // Example input matrix
    std::vector<std::vector<int>> input = {
        {21, 22, 23, 24, 25, 26, 27},
        {220, 229, 210, 211, 212, 213, 214},
        {15, 216, 217, 218, 219, 220, 221},
        {222, 255, 234, 235, 246, 247, 288},
        {229, 230, 231, 232, 233, 234, 235},
        {236, 227, 238, 239, 240, 241, 242},
        {243, 244, 245, 246, 247, 248, 249}
    };

    // Example kernel matrix
    std::vector<std::vector<int>> kernel = {
        {21, 0, 211},
        {22, 0, 212},
        {23, 0, 213}
    };

    int N = 50; // size of lfsr based random numbers / bitstream length
    uint8_t lfsr_state = 0b11011101;

    //random numbers array for SNG 
    std::vector<int> randomNumbers = LFSR_RNG_arrayGenerator(N, lfsr_state);

    StochasticTensor SCtensorInput(input, randomNumbers);
    StochasticTensor SCtensorKernel(kernel, randomNumbers);

    // Perform convolution
    std::vector<std::vector<double>> result = SC_conv2(SCtensorInput, SCtensorKernel);

    prettyPrint(result);

    // // Print the result
    // for (const auto& row : result) {
    //     for (int val : row) {
    //         std::cout << val << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}