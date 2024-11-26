#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <bitset>
#include <tuple>
#include <algorithm>
#include <cstdlib>

#include "utilities.h"

std::vector<std::vector<std::vector<uint8_t>>> convertToUint8(const std::vector<std::vector<std::vector<int>>>& input) {
    std::vector<std::vector<std::vector<uint8_t>>> output;
    output.reserve(input.size());

    for (const auto& vec2D : input) {
        std::vector<std::vector<uint8_t>> innerVec2D;
        innerVec2D.reserve(vec2D.size());

        for (const auto& vec1D : vec2D) {
            std::vector<uint8_t> innerVec1D;
            innerVec1D.reserve(vec1D.size());

            for (int value : vec1D) {
                innerVec1D.push_back(static_cast<uint8_t>(value));
            }
            innerVec2D.push_back(innerVec1D);
        }
        output.push_back(innerVec2D);
    }

    return output;
}

void prettyPrint2D(const std::vector<std::vector<double>>& matrix) {
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

void printTensor(const std::vector<std::vector<std::vector<double>>>& tensor) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        std::cout << "Tensor[" << i << "]:" << std::endl;
        for (size_t j = 0; j < tensor[i].size(); ++j) {
            std::cout << "  Tensor[" << i << "][" << j << "]: ";
            for (size_t k = 0; k < tensor[i][j].size(); ++k) {
                std::cout << tensor[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// Function to generate a vector of random floats between 0 and 1
std::vector<float> generateRandomVector(size_t size) {
    // Create a random number generator
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Define a uniform real distribution in the range [0, 1)
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    // Generate the random vector
    std::vector<float> randomVector(size);
    for (size_t i = 0; i < size; ++i) {
        randomVector[i] = dis(gen);
    }

    return randomVector;
}

std::vector<double> generateRandom1DInputs(size_t size) {
    // Create a random device and a Mersenne Twister generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define a distribution from -1 to 1
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Create a vector to hold the random numbers
    std::vector<double> randomVector(size);

    // Generate random numbers and store them in the vector
    std::generate(randomVector.begin(), randomVector.end(), [&]() { return dist(gen); });

    return randomVector;
}

std::vector<std::vector<double>> generateRandomMatrix(int NoR, int NoC) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Create the matrix
    std::vector<std::vector<double>> matrix(NoR, std::vector<double>(NoC));
    for (int i = 0; i < NoR; ++i) {
        for (int j = 0; j < NoC; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

std::vector<std::vector<double>> generateRandomInputs(int NoR, int NoC) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Create the matrix
    std::vector<std::vector<double>> matrix(NoR, std::vector<double>(NoC));
    for (int i = 0; i < NoR; ++i) {
        for (int j = 0; j < NoC; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

size_t getTotalNumberOfElements(const std::vector<std::vector<std::vector<double>>>& vec) {
    size_t count = 0;

    for (const auto& vec2D : vec) {
        for (const auto& vec1D : vec2D) {
            count += vec1D.size();
        }
    }

    return count;
}

size_t getTotalNumberOfElements2D(const std::vector<std::vector<double>>& vec) {
    size_t count = 0;

    for (const auto& innerVec : vec) {
        count += innerVec.size();
    }

    return count;
}

std::vector<std::vector<std::vector<std::vector<uint8_t>>>> create4DTensor(
    const std::vector<std::vector<std::vector<double>>>& U, const int R, const std::vector<int>& output) {
    
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> tensor;

    int index = 0;
    for (const auto& mat : U) {
        std::vector<std::vector<std::vector<uint8_t>>> innerTensorMat;
        innerTensorMat.reserve(mat.size());
        for (const auto& vec : mat) {
            std::vector<std::vector<uint8_t>> innerTensorVec;
            innerTensorVec.reserve(vec.size());
            for (size_t i = 0; i < vec.size(); ++i) {
                std::vector<uint8_t> bitstream;
                bitstream.reserve(R);
                for (size_t j = 0; j < R; ++j) {
                    bitstream.push_back(static_cast<uint8_t>(output[index++]));
                }
                innerTensorVec.push_back(bitstream);
            }
            innerTensorMat.push_back(innerTensorVec);
        }
        tensor.push_back(innerTensorMat);
    }

    return tensor;
}

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

    std::vector<float> n_forward(const std::vector<float>& inputs, 
        const std::vector<std::vector<float>>& weights, 
        const std::vector<float>& bias) {

        int input_size = weights.size();
        int output_size = bias.size();

        if (inputs.size() != input_size) {
            throw std::invalid_argument("Input size does not match the layer's input size.");
        }

        std::vector<float> outputs(output_size, 0.0);

        for (int j = 0; j < output_size; ++j) {
            for (int i = 0; i < input_size; ++i) {
                outputs[j] += inputs[i] * weights[i][j];
            }
            outputs[j] += bias[j];
        }

        return outputs;
    }

    std::vector<int8_t> castVectorToInt8(const std::vector<int>& input) {
    std::vector<int8_t> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), [](int val) {
        return static_cast<int8_t>(val);
    });
    return output;
}