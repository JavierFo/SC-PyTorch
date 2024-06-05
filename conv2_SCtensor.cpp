#include <iostream>
#include <vector>
#include <tuple>
#include "StochasticTensor.h"

std::vector<double> mult_output1;
std::vector<double> mult_output2;

std::vector<std::vector<double>> conv2(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& kernel) {
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

// std::vector<std::vector<std::vector<int>>> stochastic_Conv2(const StochasticTensor input, const StochasticTensor kernel) {
//     StochasticTensor SCtensorInput = input;
//     StochasticTensor SCtensorKernel = kernel;

//     StochasticTensor::SizeTuple inputSizes = SCtensorInput.getSize();
//     StochasticTensor::SizeTuple kernelSizes = SCtensorKernel.getSize();
    
//     const int inputRows = std::get<0>(inputSizes);
//     const int inputCols = std::get<1>(inputSizes);
//     const int kernelRows = std::get<0>(kernelSizes);
//     const int kernelCols = std::get<1>(kernelSizes);

//     const int outputRows = inputRows - kernelRows + 1;
//     const int outputCols = inputCols - kernelCols + 1;

//     std::vector<std::vector<std::vector<int>>> stochasticOutput(outputRows, std::vector<std::vector<int>>(outputCols, std::vector<int>(1, 0)));

//     // Perform 2D convolution
//     for (int i = 0; i < outputRows; ++i) {
//         for (int j = 0; j < outputCols; ++j) {
//             std::vector<int> addedScOutput(1, 0);
//             for (int ki = 0; ki < kernelRows; ++ki) {
//                 for (int kj = 0; kj < kernelCols; ++kj) {
//                     //output[i][j] += input[i + ki][j + kj] * kernel[ki][kj];
//                     std::vector<int> scMultiplication = bitstreamOperation(input.getVectorAt(i + ki, j + kj), kernel.getVectorAt(ki,kj), XNOR);
//                     addedScOutput = bitstreamOperation(scMultiplication, addedScOutput, MUX);
//                 }
//             }
//             stochasticOutput[i][j] = addedScOutput;
//         }
//     }
//     return stochasticOutput;
// }

// std::vector<std::vector<double>> polar_Conv2(const StochasticTensor input, const StochasticTensor kernel) {
//     StochasticTensor SCtensorInput = input;
//     StochasticTensor SCtensorKernel = kernel;

//     StochasticTensor::SizeTuple inputSizes = SCtensorInput.getSize();
//     StochasticTensor::SizeTuple kernelSizes = SCtensorKernel.getSize();
    
//     const int inputRows = std::get<0>(inputSizes);
//     const int inputCols = std::get<1>(inputSizes);
//     const int kernelRows = std::get<0>(kernelSizes);
//     const int kernelCols = std::get<1>(kernelSizes);

//     const int outputRows = inputRows - kernelRows + 1;
//     const int outputCols = inputCols - kernelCols + 1;

//     // Create the output matrix with appropriate dimensions
//     std::vector<std::vector<double>> output(outputRows, std::vector<double>(outputCols, 0));

//     // Perform 2D convolution
//     for (int i = 0; i < outputRows; ++i) {
//         for (int j = 0; j < outputCols; ++j) {
//             //std::vector<int> addedScOutput(1, 0);
//             double acumulatedAdder = 0;
//             for (int ki = 0; ki < kernelRows; ++ki) {
//                 for (int kj = 0; kj < kernelCols; ++kj) {
//                     //output[i][j] += input[i + ki][j + kj] * kernel[ki][kj];
//                     //std::vector<int> scMultiplication = bitstreamOperation(input.getVectorAt(i + ki, j + kj), kernel.getVectorAt(ki,kj), XNOR);
//                     double scMultiplication = calculatePx(input.getVectorAt(i + ki, j + kj), BIPOLAR)*calculatePx(kernel.getVectorAt(ki,kj), BIPOLAR);
//                     acumulatedAdder = (1-0.5)*scMultiplication + 0.5*acumulatedAdder;
//                     //addedScOutput = scMultiplication;
//                 }
//             }
//             output[i][j] = acumulatedAdder*255000;
//         }
//     }
//     return output;
// }

std::vector<int> concatenateVectors(const std::vector<int>& vector1, const std::vector<int>& vector2) {
    if (vector2.size() > 1 && std::any_of(vector2.begin(), vector2.end(), [](int i) { return i == 1; })) {
        std::vector<int> result = vector1;
        result.insert(result.end(), vector2.begin(), vector2.end());
        return result;
    }
    return vector1;
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<double>>> SC_conv2 (const StochasticTensor input, const StochasticTensor kernel) { 
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

    const double inputBitstreamSize = std::get<2>(inputSizes);

    std::vector<std::vector<double>> output(outputRows, std::vector<double>(outputCols, 0));
    std::vector<std::vector<std::vector<int>>> stochasticOutput(outputRows, std::vector<std::vector<int>>(outputCols, std::vector<int>(1, 0)));
    std::vector<std::vector<double>> stochasticOutputWthAcc(outputRows, std::vector<double>(outputCols, 0));

    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            double acumulatedAdder = 0;
            std::vector<int> addedScOutput(1, 0);
            std::vector<int> addedScOutput_acc(1, 0);

            for (int ki = 0; ki < kernelRows; ++ki) {
                for (int kj = 0; kj < kernelCols; ++kj) {
                    //Polar Multiplication and addition with formula: Pq = (1-Ps)(Pa) + (Ps)(Pb), where Ps = 0.5
                    //For polar Multiplication, input and kernel stochastic bitstreams are converted to polar probabilities.
                    double scPolarMultiplication = calculatePx(input.getVectorAt(i + ki, j + kj), BIPOLAR)*calculatePx(kernel.getVectorAt(ki,kj), BIPOLAR);
                    acumulatedAdder = (1-0.5)*scPolarMultiplication + 0.5*acumulatedAdder;
                    //Bitwise Multiplication with XNOR and addition with both MUX and vector accumulator.
                    //The vector accumulator works by concatenating the result of the mult with the previous stochastic vector, 
                    //then, the number of 1s in the final stochastic bitstream are counted and divided by the accumulated vector's size. 
                    std::vector<int> scMultiplication = bitstreamOperation(input.getVectorAt(i + ki, j + kj), kernel.getVectorAt(ki,kj), XNOR);
                    addedScOutput = bitstreamOperation(scMultiplication, addedScOutput, MUX);
                    addedScOutput_acc = concatenateVectors(scMultiplication,addedScOutput_acc);
                }
            }
            double scale = addedScOutput_acc.size()/inputBitstreamSize;
            output[i][j] = acumulatedAdder*scale;
            stochasticOutput[i][j] = addedScOutput;
            stochasticOutputWthAcc[i][j] = calculatePx(addedScOutput_acc, BIPOLAR)*scale;
        }
    }
    return std::make_tuple(output, stochasticOutput, stochasticOutputWthAcc);
}

                    //mult_output1.push_back(scPolarMultiplication*1000);
                    //mult_output2.push_back(calculatePx(scMultiplication, BIPOLAR)*1000);
                    //std::cout << "a: " << scPolarMultiplication << "b: " << calculatePx(scMultiplication, UNIPOLAR) << std::endl;
                    
                    //std::cout << "acumulatedAdder: " << acumulatedAdder << std::endl;
                    //std::cout << "vec size: " << addedScOutput_acc.size() << std::endl;
                    //std::cout << "no. 1s: " << std::count(addedScOutput_acc.begin(), addedScOutput_acc.end(), 1) << std::endl;

            //double scale = addedScOutput_acc.size()/inputBitstreamSize;
            //std::cout << "iteration" << j << std::endl;

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

std::vector<std::vector<double>> toRealTensor(const std::vector<std::vector<std::vector<int>>>& tensor, int scale, BitstreamRepresentation mode) {
    std::vector<std::vector<double>> result;

    for (const auto& matrix : tensor) {
        std::vector<double> rowResult;
        for (const auto& vec : matrix) {
            int countOnes = 0;
            for (int val : vec) {
                if (val == 1) {
                    countOnes++;
                }
            }
            double probability = static_cast<double>(countOnes) / vec.size();
                if (mode == UNIPOLAR) {
                    rowResult.push_back(probability * scale);
                } else if (mode == BIPOLAR) {
                    rowResult.push_back(((2 * probability) - 1)* scale);
                    // double bipolarValue = ((2 * probability) - 1);
                    // double realNumber = (bipolarValue + 1) / 2 * (255 - (-255)) + (-255);
                    // rowResult.push_back(realNumber);
                } else {
                    throw std::invalid_argument("Invalid mode");
                }
        }
        result.push_back(rowResult);
    }
    return result;
}

std::vector<std::vector<double>> realTensorNormalizer (const std::vector<std::vector<double>>& matrix, int minVal, int maxVal) {
    std::vector<std::vector<double>> result;
    for (const auto& row : matrix) {
        std::vector<double> rowResult;
        for (const auto& value : row) {
            double normalizedValue = normalizeRealNumber(value, minVal, maxVal);
            rowResult.push_back(normalizedValue);
        }
        result.push_back(rowResult);
    }
    return result;
}

int main() {
    // Example input matrix
    std::vector<std::vector<double>> input = {
        {-5, 150, 80},
        {250, 0, -180},
        {-220, 50, 205}
    };
    std::vector<std::vector<double>> kernel = {
        {-105, -250},
        {-30, -3}
    };

    // std::vector<std::vector<double>> sc_input = {
    //     {.1, -.2, .3},
    //     {.4, .5, -.6},
    //     {.7, -.8, .9}
    // };
    // std::vector<std::vector<double>> sc_kernel = {
    //     {1, 0},
    //     {-1, -.5}
    // };

    std::vector<std::vector<double>> sc_input = realTensorNormalizer(input,0,255);
    std::vector<std::vector<double>> sc_kernel = realTensorNormalizer(kernel,0,255);

    int N = 1000000; // size of lfsr based random numbers / bitstream length

    // prettyPrint(sc_input);
    // prettyPrint(sc_kernel);

    StochasticTensor SCtensorInput(sc_input, N, MT19937, BIPOLAR);
    StochasticTensor SCtensorKernel(sc_kernel, N, MT19937, BIPOLAR);

    //std::vector<std::vector<double>> realSCtensorInput = SCtensorInput.toRealTensor(1, BIPOLAR);
    //std::vector<std::vector<double>> realSCtensorKernel = SCtensorKernel.toRealTensor(1, BIPOLAR);
    // std::cout << "SC_Input:";
    // prettyPrint(realSCtensorInput);
    // std::cout << "SC_Kernel:";
    // prettyPrint(realSCtensorKernel);

    std::cout << "1. Convolution wth Original function:";
    std::vector<std::vector<double>> originalResult = conv2(input, kernel);
    prettyPrint(originalResult);

    // std::cout << "Convolution wth Original function but Stochastic input_kernel:";
    // std::vector<std::vector<double>> realSCresult = conv2(realSCtensorInput, realSCtensorKernel);
    // prettyPrint(realSCresult);

    auto scConvOutput = SC_conv2(SCtensorInput, SCtensorKernel);

    auto double2DVector = std::get<0>(scConvOutput);
    auto int3DVector = std::get<1>(scConvOutput);
    auto acc_double2DVector = std::get<2>(scConvOutput);

    // std::cout << "Convolution wth polar mult and Pq = (1-Ps)(Pa) + (Ps)(Pb) addition:";
    // //std::vector<std::vector<double>> result = SC_conv2(SCtensorInput, SCtensorKernel);
    // prettyPrint(double2DVector);

    // std::cout << "Convolution wth Stochastic Vectors:";
    // //std::vector<std::vector<std::vector<int>>> scConvOutput = SC_conv2(SCtensorInput, SCtensorKernel);
    // prettyPrint(toRealTensor(int3DVector, 4, BIPOLAR));

    std::cout << "1. Stochastic Convolution (bitstream length 1,000,000):";
    prettyPrint(acc_double2DVector);

    // StochasticTensor SCtensorInput0(sc_input, 2000, MT19937, BIPOLAR);
    // StochasticTensor SCtensorKernel0(sc_kernel, 2000, MT19937, BIPOLAR);

    // auto scConvOutput0 = SC_conv2(SCtensorInput, SCtensorKernel);
    // auto acc_double2DVector0 = std::get<2>(scConvOutput0);

    // std::cout << "1. Stochastic Convolution (bitstream length 2,000):";
    // prettyPrint(acc_double2DVector0);

    // std::vector<std::vector<double>> sc_input2 = {
    //     {0, -.212, .38},
    //     {-.4009, -.55, -.66},
    //     {-.725, -.8125, -.9725}
    // };
    // std::vector<std::vector<double>> sc_kernel2 = {
    //     {.99, 0},
    //     {-.11, -.5}
    // };

    // std::cout << "\n";
    // std::cout << "2. Convolution wth Original function :";
    // std::vector<std::vector<double>> originalResult2 = conv2(sc_input2, sc_kernel2);
    // prettyPrint(originalResult2);

    // StochasticTensor SCtensorInput2(sc_input2, 500, MT19937, BIPOLAR);
    // StochasticTensor SCtensorKernel2(sc_kernel2, 500, MT19937, BIPOLAR);

    // auto scConvOutput2 = SC_conv2(SCtensorInput2, SCtensorKernel2);

    // auto double2DVector2 = std::get<0>(scConvOutput2);
    // auto int3DVector2 = std::get<1>(scConvOutput2);
    // auto acc_double2DVector2 = std::get<2>(scConvOutput2);

    // std::cout << "2. Stochastic Convolution (bitstream length 500):";
    // prettyPrint(acc_double2DVector2);

    // std::vector<std::vector<double>> sc_input3 = {
    //     {0, -.735, -.377},
    //     {-.411, 0, -.665},
    //     {.733, -1, 0}
    // };
    // std::vector<std::vector<double>> sc_kernel3 = {
    //     {-1, 0},
    //     {-1, -.22}
    // };

    // std::cout << "\n";
    // std::cout << "3. Convolution wth Original function :";
    // std::vector<std::vector<double>> originalResult3 = conv2(sc_input3, sc_kernel3);
    // prettyPrint(originalResult3);

    // StochasticTensor SCtensorInput3(sc_input3, 255, LFSR, BIPOLAR);
    // StochasticTensor SCtensorKernel3(sc_kernel3, 255, LFSR, BIPOLAR);

    // auto scConvOutput3 = SC_conv2(SCtensorInput3, SCtensorKernel3);

    // auto double2DVector3 = std::get<0>(scConvOutput3);
    // auto int3DVector3 = std::get<1>(scConvOutput3);
    // auto acc_double2DVector3 = std::get<2>(scConvOutput3);

    // std::cout << "2. Stochastic Convolution (LFSR, bitstream length 255):";
    // prettyPrint(acc_double2DVector3);
    
    return 0;
}





    // std::vector<std::vector<int>> input = {
    //     {0, -22, 7, -24, 25, 2, -255},
    //     {220, -229, 210, 211, -212, 213, 214},
    //     {15, 216, -217, 218, 219, -220, 221},
    //     {-122, 155, 134, -135, 146, 147, -188},
    //     {229, 230, 231, 232, 233, 234, 235},
    //     {236, -227, 238, 239, 240, 241, -242},
    //     {255, 32, -9, 44, 15, -16, 0}
    // };

    // // Example kernel matrix
    // std::vector<std::vector<int>> kernel = {
    //     {10, -10, 10},
    //     {-10, 0, 10},
    //     {10, 10, -10}
    // };




// // Function to convert a bipolar stochastic bitstream to its real form
// double convertBipolarStochasticToReal(const std::vector<int>& bitstream) {
//     // Calculate the probability of 1s
//     double countOnes = std::count(bitstream.begin(), bitstream.end(), 1);
//     double probability = countOnes / bitstream.size();

//     // Convert probability to bipolar value
//     double bipolarValue = 2 * probability - 1;

//     return bipolarValue;
// }

// // Function to calculate the real number from a stochastic bitstream
// double convertFromStochasticBitstream(const std::vector<int>& bitstream, double minRange, double maxRange) {
//     // Calculate the probability
//     double countOnes = std::count(bitstream.begin(), bitstream.end(), 1);
//     double probability = countOnes / bitstream.size();

//     // Convert probability to bipolar value
//     double bipolarValue = 2 * probability - 1;

//     // Denormalize to the original range
//     double realNumber = (bipolarValue + 1) / 2 * (maxRange - minRange) + minRange;

//     return realNumber;
// }