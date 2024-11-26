// // ScCudaTorch.cpp
#include <iostream>
#include <vector>
#include <cstdio>
#include <random>
#include <cstdint>
#include <vector>

#include "utilities.h"

#include "pcg_random.hpp"
#include "xorshift.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h> 
#include <pybind11/chrono.h>

// Function prototype
std::vector<std::vector<double>> ScCudaConv2d(
    const std::vector<std::vector<double>>& polarInputData,
    const std::vector<std::vector<double>>& polarKernelData,
    const std::vector<std::vector<double>>& randomMatrix_Input,
    const std::vector<std::vector<double>>& randomMatrix_Kernel);

std::vector<float> ScCudaFcLayer(
    const std::vector<float>& inputs, 
    const std::vector<std::vector<float>>& weights, 
    const std::vector<float>& biases, 
    const std::vector<std::vector<float>>& randomMatrix_input, 
    const std::vector<std::vector<float>>& randomMatrix_weights, 
    const std::vector<std::vector<float>>& randomMatrix_biases, 
    const int num_Outputs);

enum RandomGeneratorType {
    MT19937,
    LFSR_16,
    PCG,
    XORSHIFT
};

class LFSR16 {
public:
    // Constructor: Initializes the LFSR with a seed
    LFSR16(uint16_t seed) : state(seed) {}

    // Function to advance the LFSR and produce the next random number
    uint16_t next() {
        // Taps: 16, 15, 13, 4 (represented by bits 15, 14, 12, and 3 in 0-based indexing)
        // Polynomial: x^16 + x^15 + x^13 + x^4 + 1 => Tap mask: 0xD008
        bool bit = ((state >> 15) ^ (state >> 14) ^ (state >> 12) ^ (state >> 3)) & 1;
        state = (state << 1) | bit; // Shift and add the feedback bit
        return state;               // Return the next value
    }

    // Function to generate a random number as required by std::uniform_real_distribution
    uint16_t operator()() {
        return next();
    }

    // Min and max values to satisfy std::uniform_real_distribution's requirements
    static constexpr uint16_t min() { return 0; }
    static constexpr uint16_t max() { return std::numeric_limits<uint16_t>::max(); }

private:
    uint16_t state; // The LFSR state
};

std::vector<std::vector<double>> generate_random_matrix(int R, int C, RandomGeneratorType generator_type) {
    std::random_device rd;  // Random device for generating seeds
    std::vector<std::vector<double>> matrix(R, std::vector<double>(C));
    // Define a uniform real distribution in the range [0.0, 1.0)
    std::uniform_real_distribution<double> dis(0.0, 1.0);

switch (generator_type) {
        case MT19937: {
            for (int i = 0; i < R; ++i) {
                uint32_t seed = rd();  // Generate a new seed for each row
                std::mt19937 gen(seed);  // Mersenne Twister random number generator
                for (int j = 0; j < C; ++j) {
                    matrix[i][j] = dis(gen);
                }
            }
            break;
        }
        case LFSR_16: {
            for (int i = 0; i < R; ++i) {
                uint16_t seed = rd() & 0xFFFF;
                LFSR16 lfsr(seed);
                for (int j = 0; j < C; ++j) {
                    matrix[i][j] = dis(lfsr);
                }
            }
            break;
        }
        case PCG: {
           for (int i = 0; i < R; ++i) {
                pcg_extras::seed_seq_from<std::random_device> seed_source;
                pcg32 rng(seed_source);
                for (int j = 0; j < C; ++j) {
                    matrix[i][j] = dis(rng);
                }
            }
            break;
        }
        case XORSHIFT: {
            for (int i = 0; i < R; ++i) {
                uint32_t seed32 = rd();
                using rng32_type = xorshift_detail::xorshiftstar<xorshift32plain32a, uint16_t, 0xb2e1cb1dU>;
                rng32_type rng32(seed32);
                for (int j = 0; j < C; ++j) {
                    matrix[i][j] = dis(rng32);
                }
            }
            break;
        }
        default:
            std::cerr << "Invalid generator type" << std::endl;
    }
    return matrix;
}

PYBIND11_MODULE(sc_cuda_torch, m) {
    pybind11::enum_<RandomGeneratorType>(m, "RandomGeneratorType")
        .value("MT19937", RandomGeneratorType::MT19937)
        .value("LFSR_16", RandomGeneratorType::LFSR_16)
        .value("PCG", RandomGeneratorType::PCG)
        .value("XORSHIFT", RandomGeneratorType::XORSHIFT)
        .export_values();

    m.def("ScCudaFcLayer", &ScCudaFcLayer, "A function that does ScCudaFcLayer with weights, bias and input using CUDA");
    m.def("ScCudaConv2d", &ScCudaConv2d, "A function that does ScConv2D an input and kernel using CUDA");
    m.def("generate_random_matrix", &generate_random_matrix, "A function that generates_random_matrix with different types of RNG");
}

// template <typename T>
// std::vector<double> flatten_(const std::vector<std::vector<T>>& input) {
//     std::vector<double> output;
//     for (const auto& vec : input) {
//         for (const auto& val : vec) {
//             output.push_back(static_cast<double>(val));
//         }
//     }
//     return output;
// }

// std::vector<float> double_to_float_vector(const std::vector<double>& double_vector) {
//     std::vector<float> float_vector(double_vector.size());
//     std::transform(double_vector.begin(), double_vector.end(), float_vector.begin(),
//                    [](double d) { return static_cast<float>(d); });
//     return float_vector;
// }

// std::vector<std::vector<float>> double_to_float_matrix(const std::vector<std::vector<double>>& float_matrix) {
//     std::vector<std::vector<float>> double_matrix;
//     std::transform(float_matrix.begin(), float_matrix.end(), std::back_inserter(double_matrix), [](const std::vector<double>& row) {
//         return std::vector<float>(row.begin(), row.end());
//     });
//     return double_matrix;
// }

// int main() {
//     // // Example input data
//     // std::vector<std::vector<double>> polarInputData = {{1.0, 2.0}, {3.0, 4.0}};
//     // std::vector<std::vector<double>> polarKernelData = {{0.5, 0.5}, {0.5, 0.5}};
//     // std::vector<std::vector<double>> randomMatrix_Input = {{0.1, 0.2}, {0.3, 0.4}};
//     // std::vector<std::vector<double>> randomMatrix_Kernel = {{0.1, 0.2}, {0.3, 0.4}};
//     // // Call the ScCudaConv2d function
//     // std::vector<std::vector<double>> result = ScCudaConv2d(polarInputData, polarKernelData, randomMatrix_Input, randomMatrix_Kernel);

//     // std::vector<std::vector<double>> sc_input = {
//     //     {.1, -.2, .3},
//     //     {.4, .5, -.6},
//     //     {.7, -.8, .9}
//     // };
//     // std::vector<std::vector<double>> sc_kernel = {
//     //     {1, 0},
//     //     {-1, -.5}
//     // };
    
//     // size_t bitstreamLength = 5000; // Size of the vector
//     // // Generate random matrix
//     // size_t totalElements1 = getTotalNumberOfElements2D(sc_input);
//     // std::vector<std::vector<double>> randomMatrix1 = generateRandomMatrix(totalElements1, bitstreamLength);
    
//     // size_t totalElements2 = getTotalNumberOfElements2D(sc_kernel);
//     // std::vector<std::vector<double>> randomMatrix2 = generateRandomMatrix(totalElements2, bitstreamLength);
    
//     // std::vector<std::vector<double>> result = ScCudaConv2d(sc_input, sc_kernel, randomMatrix1, randomMatrix2);

//     // // Print the result
//     // std::cout << "Result:" << std::endl;
//     // for (const auto& row : result) {
//     //     for (const auto& val : row) {
//     //         std::cout << val << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }



//     // std::vector<std::vector<double>> weights = {
//     //     {-0.1, -0.2, -0.3, -0.4, -0.5, -0.6},
//     //     {-0.2, -0.3, -0.4, -0.5, -0.6, -0.7},
//     //     {0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
//     //     {0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
//     //     {0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
//     //     {0.6, 0.7, 0.8, 0.9, 1.0, 0.1},
//     //     {0.7, 0.8, 0.9, 1.0, 0.1, 0.2},
//     //     {0.8, 0.9, 1.0, 0.1, 0.2, 0.3},
//     //     {0.9, 1.0, 0.1, 0.2, 0.3, 0.4},
//     //     {1.0, 0.1, 0.2, 0.3, 0.4, 0.5},
//     //     {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
//     //     {0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
//     //     {0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
//     //     {0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
//     //     {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
//     //     };
    
//     //     std::vector<double> bias = {-0.1, -0.2, -0.3, -0.4, -0.5, -0.6};
    
//     //     std::vector<double> inputs = {-0.5, -0.4, -0.3, -0.2, -0.1, -0.6, -0.7, -0.8, -0.9, -1.0, 0.9, 0.8, 0.7, 0.6, 0.5};

//     size_t sizeI = 1000;
//     std::vector<float> inputs = double_to_float_vector(generateRandom1DInputs(sizeI));   
    
//     size_t sizeB = 300;
//     std::vector<float> bias = double_to_float_vector(generateRandom1DInputs(sizeB));

//     int inputsSize = sizeI; int OutputsSize = sizeB;
//     std::vector<std::vector<float>> weights = double_to_float_matrix(generateRandomMatrix(inputsSize, OutputsSize));


//     int bit_length = 1000;
//     std::cout << " ";
//     std::cout << "Normal NN Values: ";
//     std::cout << "\n";
//     std::vector<float> normalForward = n_forward(inputs, weights, bias);
//     for (int j = 0; j < normalForward.size(); ++j) {
//         std::cout << normalForward[j] << " ";
//     }
//     std::cout << "\n";

//     int weightsSize = inputs.size() * bias.size();
//     std::vector<float> flattenedWeights = double_to_float_vector(flatten_(weights));
//     std::vector<std::vector<float>> randomMatrixInput = double_to_float_matrix(generateRandomMatrix(inputs.size(), bit_length));
//     std::vector<std::vector<float>> randomMatrixWeights = double_to_float_matrix(generateRandomMatrix(weightsSize, bit_length));
//     std::vector<std::vector<float>> randomMatrixBias = double_to_float_matrix(generateRandomMatrix(bias.size(), bit_length));

//     std::vector<float> h_output = ScCudaFcLayer(inputs, flattenedWeights, bias, randomMatrixInput, randomMatrixWeights, randomMatrixBias, OutputsSize);
//     std::cout << "\n"; std::cout << "\n";
//     std::cout << "SC forward output:" << std::endl;
//     for (int j = 0; j < h_output.size(); ++j) {
//         std::cout << h_output[j] << " ";
//     }

//     std::vector<float> h_output2 = ScCudaFcLayer(inputs, flattenedWeights, {}, randomMatrixInput, randomMatrixWeights, {}, OutputsSize);
//     std::cout << "\n"; std::cout << "\n";
//     std::cout << "SC forward output WITHOUT BIAS:" << std::endl;
//     for (int j = 0; j < h_output2.size(); ++j) {
//         std::cout << h_output2[j] << " ";
//     }

//     return 0;
// }
