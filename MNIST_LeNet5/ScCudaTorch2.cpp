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
std::vector<std::vector<float>> ScCudaConv2d(
    const std::vector<float>& polarInputData,
    const std::vector<float>& polarKernelData,
    const std::vector<float>& randomMatrix_Input,
    const std::vector<float>& randomMatrix_Kernel,
    const int bitstream_Length,
    const int height,
    const int width,
    const int heightK);

std::vector<float> ScCudaFcLayer(
    const std::vector<float>& inputs, 
    const std::vector<float>& weights, 
    const std::vector<float>& biases, 
    const std::vector<float>& randomMatrix_input, 
    const std::vector<float>& randomMatrix_weights, 
    const std::vector<float>& randomMatrix_biases, 
    const int num_Outputs,
    const int bitstream_Length);

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

std::vector<float> generate_random_matrix(int R, int C, RandomGeneratorType generator_type) {
    std::random_device rd;  // Random device for generating seeds
    //std::vector<std::vector<double>> matrix(R, std::vector<double>(C));
    std::vector<float> matrix1D(R * C);
    // Define a uniform real distribution in the range [0.0, 1.0)
    std::uniform_real_distribution<float> dis(0.0, 1.0);

switch (generator_type) {
        case MT19937: {
            // for (int i = 0; i < R; ++i) {
            //     uint32_t seed = rd(); 
            //     std::mt19937 gen(seed); 
            //     for (int j = 0; j < C; ++j) {
            //         matrix[i][j] = dis(gen);
            //     }
            // }
            // break;
            uint32_t seed = rd(); 
            std::mt19937 gen(seed); 
            for (int i = 0; i < R*C; ++i) {
                matrix1D[i] = dis(gen);
            }
            break;
        }
        case LFSR_16: {
            for (int i = 0; i < R; ++i) {
                uint16_t seed = rd() & 0xFFFF;
                LFSR16 lfsr(seed);
                for (int j = 0; j < C; ++j) {
                    matrix1D[i * C + j] = dis(lfsr);
                }
            }
            break;
        }
        case PCG: {
        //    for (int i = 0; i < R; ++i) {
        //         pcg_extras::seed_seq_from<std::random_device> seed_source;
        //         pcg32 rng(seed_source);
        //         for (int j = 0; j < C; ++j) {
        //             matrix1D[i * C + j] = dis(rng);
        //         }
        //     }
            pcg_extras::seed_seq_from<std::random_device> seed_source; 
            pcg32 rng(seed_source); 
            for (int i = 0; i < R*C; ++i) {
                matrix1D[i] = dis(rng);
            }
            break;
        }
        case XORSHIFT: {
            // for (int i = 0; i < R; ++i) {
            //     uint32_t seed32 = rd();
            //     using rng32_type = xorshift_detail::xorshiftstar<xorshift32plain32a, uint16_t, 0xb2e1cb1dU>;
            //     rng32_type rng32(seed32);
            //     for (int j = 0; j < C; ++j) {
            //         matrix1D[i * C + j] = dis(rng32);
            //     }
            // }
            uint32_t seed32 = rd();
            using rng32_type = xorshift_detail::xorshiftstar<xorshift32plain32a, uint16_t, 0xb2e1cb1dU>;
            rng32_type rng32(seed32);
            for (int i = 0; i < R*C; ++i) {
                matrix1D[i] = dis(rng32);
            }
            break;
        }
        default:
            std::cerr << "Invalid generator type" << std::endl;
    }
    return matrix1D;
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