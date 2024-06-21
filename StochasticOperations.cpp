#include <iostream>
#include "ScTorch.h"
#include "StochasticTensor.h"
#include <random>
#include <iostream>
#include <vector>

std::vector<std::vector<double>> conv2_(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& kernel) {
    int inputRows = input.size();
    int inputCols = input[0].size();
    int kernelRows = kernel.size();
    int kernelCols = kernel[0].size();

    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;

    // Create the output matrix with appropriate dimensions
    std::vector<std::vector<double>> output(outputRows, std::vector<double>(outputCols, 0));

    // Perform 2D convolution
    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            double acumulator = 0;
            // Compute the convolution sum for the current position
            for (int ki = 0; ki < kernelRows; ++ki) {
                for (int kj = 0; kj < kernelCols; ++kj) {
                   double mult = input[i + ki][j + kj] * kernel[ki][kj];
                   acumulator = mult + acumulator;
                    //std::cout << output[i][j] << "\t";
                }
            }
            output[i][j] = acumulator;
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

std::vector<double> createAlternatingVector(int size) {
    std::vector<double> vec(size);
    std::vector<double> values = {0.5, 0.3, 0.2, 0.1, 0.8};
    int values_size = values.size();

    for (int i = 0; i < size; ++i) {
        vec[i] = values[i % values_size];
    }

    return vec;
}

std::vector<std::vector<double>> createAlternatingMatrix(int width, int height) {
    std::vector<std::vector<double>> matrix(height, std::vector<double>(width));
    std::vector<double> values = {0.5, 0.3, 0.2, 0.1, 0.8};
    int values_size = values.size();

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            matrix[i][j] = values[(i * width + j) % values_size];
        }
    }

    return matrix;
}

int main() {
    // std::vector<std::vector<double>> sc_input = {
    //     {.1, -.2, .3},
    //     {.4, .5, -.6},
    //     {.7, -.8, .9}
    // };
    // std::vector<std::vector<double>> sc_kernel = {
    //     {1, 0},
    //     {-1, -.5}
    // };

    // int N = 90000; // size of lfsr based random numbers / bitstream length

    // StochasticTensor SCtensorInput(sc_input, N, MT19937, BIPOLAR);
    // StochasticTensor SCtensorKernel(sc_kernel, N, MT19937, BIPOLAR);

    // std::cout << "1. Convolution wth Original function:";
    // std::vector<std::vector<double>> originalResult = conv2_(sc_input, sc_kernel);
    // prettyPrint(originalResult);

    // auto acc_double2DVector = ScConv2d(SCtensorInput, SCtensorKernel, 1, 1, 1);
    // std::cout << "1. Stochastic Convolution:";
    // prettyPrint(acc_double2DVector);

//     std::vector<std::vector<std::vector<double>>> input = 
//   {
//     {
//         {.1, .2, .3, .4, .5},
//         {.6, .7, .8, .9, .10},
//         {.11, .12, .13, .14, .15}
//     },
//     {
//         {.26, .27, .28, .29, .30},
//         {.31, .32, .33, .34, .35},
//         {.36, .37, .38, .39, .40}
//     },
//     {
//         {.51, .52, .53, .54, .55},
//         {.56, .57, .58, .59, .60},
//         {.61, .62, .63, .64, .65}
//     }
// };

//    std::vector<std::vector<std::vector<double>>> kernel = 
// {
//     {
//         {1, 0, -1},
//         {1, 0, -1},
//         {1, 0, -1}
//     },
//     {
//         {0, 1, 0},
//         {0, 1, 0},
//         {0, 1, 0}
//     },
//     {
//         {-1, -1, -1},
//         {0, 0, 0},
//         {1, 1, 1}
//     }
// };

//   int padding = 2;
//   int stride = 2;
//   int dilation = 1;

//   int N = 1000;

//   StochasticTensor SCtensorInput(input, N, MT19937, BIPOLAR);
//   StochasticTensor SCtensorKernel(kernel, N, MT19937, BIPOLAR);

//   std::vector<std::vector<std::vector<double>>> output = ScConv3d(SCtensorInput, SCtensorKernel, padding, stride, dilation);

//   // Print the output (adjust for your data dimensions)
//   for (int d = 0; d < output.size(); ++d) {
//     for (int h = 0; h < output[d].size(); ++h) {
//       for (int w = 0; w < output[d][h].size(); ++w) {
//         std::cout << output[d][h][w] << " ";
//       }
//       std::cout << std::endl;
//     }
//     std::cout << std::endl;
//   }

    std::vector<std::vector<double>> weights = {
    {-0.1, 0.2, 0.3, 0.4, -0.5},
    {-0.2, 0.3, 0.4, 0.5, -0.6},
    {-0.3, 0.4, 0.5, 0.6, -0.7},
    {-0.4, 0.5, 0.6, 0.7, -0.8},
    {-0.5, 0.6, 0.7, 0.8, -0.9},
    {-0.6, 0.7, 0.8, 0.9, -1.0},
    {-0.7, 0.8, 0.9, 1.0, -0.1},
    {-0.8, 0.9, 1.0, 0.1, -0.2},
    {-0.9, 1.0, 0.1, 0.2, -0.3},
    {-1.0, 0.1, 0.2, 0.3, -0.4},
    {-0.1, 0.2, 0.3, 0.4, -0.5},
    {-0.2, 0.3, 0.4, 0.5, -0.6}
    };

    std::vector<double> bias = {-0.9, 0.2, 0.3, 0.4, -0.5};

    std::vector<double> inputs = {-0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, -0.8};

    int N = 255;

    //FullyConnectedLayer layer1(weights, bias);
    ScFcLayer scLayer1(weights, bias, N, MT19937, BIPOLAR);

    // Forward pass without activation
    std::vector<double> outputs1 = scLayer1.forward(inputs);
    std::cout << "Outputs without activation: ";
    for (double val : outputs1) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    int size = 128; // Example size
    ScFcLayer scLayer2(size, 10, N, MT19937, BIPOLAR);

    std::vector<double> inputs_ = createAlternatingVector(size);
    //Forward pass with activation
    std::vector<double> outputs2 = scLayer2.forward(inputs_);
    std::cout << "Outputs with activation: ";
    for (double val : outputs2) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // std::vector<double> outputs3 = scLayer2.sigmoid(inputs_);
    // std::cout << "Outputs with activation: ";
    // for (double val : outputs3) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

    // std::vector<double> outputs4 = scLayer2.tanh(inputs_);
    // std::cout << "Outputs with activation: ";
    // for (double val : outputs4) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

    // std::vector<double> outputs5 = scLayer2.softmax(inputs_);
    // std::cout << "Outputs with activation: ";
    // for (double val : outputs5) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

    return 0;
}
