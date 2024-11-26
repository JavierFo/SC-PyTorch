#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "utilities.h"

__global__ void stochasticTensorGenerator(const float* inputData, const float* randomMatrix, int8_t* output, int inputData_size, int RM_cols);

#define cudaCheckError(err) (cudaCheck(err, __FILE__, __LINE__))
inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Helper function to flatten a 2D vector into a 1D vector
std::vector<float> flatten2D(const std::vector<std::vector<double>>& input) {
    std::vector<float> output;
    for (const auto& vec : input) {
        for (const auto& val : vec) {
            output.push_back(static_cast<float>(val));
        }
    }
    return output;
}

template <typename T>
std::vector<float> flatten(const std::vector<std::vector<T>>& input) {
    std::vector<float> output;
    for (const auto& vec : input) {
        for (const auto& val : vec) {
            output.push_back(static_cast<float>(val));
        }
    }
    return output;
}

// stochasticTensorGenerator
__global__ void stochasticTensorGenerator(const float* inputData, const float* randomMatrix, int8_t* output, int inputData_size, int RM_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < inputData_size * RM_cols) {
        int n_idx = idx / RM_cols;
        int rm_col_idx = idx % RM_cols;
        int rm_idx = n_idx * RM_cols + rm_col_idx;
        
        // Determine if inputData[n_idx] is positive or negative
        if (inputData[n_idx] > 0) {
            // Add 1 to all positions separated by RM_cols
            if (rm_col_idx == 0) {
                output[idx] = 1;
            } else {
                output[idx] = (randomMatrix[rm_idx] < (inputData[n_idx])) ? 1 : 0;
            }
        } else if (inputData[n_idx] < 0) {
            // Add -1 to all positions separated by RM_cols
            if (rm_col_idx == 0) {
                output[idx] = -1;
            } else {
                output[idx] = (randomMatrix[rm_idx] < ((inputData[n_idx] * -1))) ? 1 : 0;
            }
        } else if (inputData[n_idx] == 0) {
            output[idx] = 0;
        } 
    }
}

__constant__ int inputWidthConv2;
__constant__ int KernelHeightConv2;
__constant__ int KernelWidthConv2;
__constant__ int OutputHeightConv2;
__constant__ int OutputWidthConv2;
__constant__ int NConv2;

// CUDA kernel for 2D convolution
__global__ void conv2D(const int8_t* input, const int8_t* kernel, float* output) {
    //extern __shared__ int shared_result[];
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < OutputHeightConv2 && j < OutputWidthConv2) {
        float accumulatedOnes_pos = 0;
        float accumulatedOnes_neg = 0;
        int numberOfAccumulations = 0;
        for (size_t m = 0; m < KernelHeightConv2; ++m) {
            for (size_t n = 0; n < KernelWidthConv2; ++n) {
                numberOfAccumulations++;
                //count = 0;
                if((input[((i + m) * inputWidthConv2 * NConv2 + (j + n) * NConv2)]) * (kernel[(m * KernelWidthConv2 * NConv2 + n * NConv2)]) == 1){
                    for (size_t bit_counter = 1; bit_counter < NConv2; ++bit_counter) {
                        if(((input[((i + m) * inputWidthConv2 * NConv2 + (j + n) * NConv2)+bit_counter] & kernel[(m * KernelWidthConv2 * NConv2 + n * NConv2)+bit_counter])) == 1)
                        {accumulatedOnes_pos++;}
                    }
                } else if ((input[((i + m) * inputWidthConv2 * NConv2 + (j + n) * NConv2)]) * (kernel[(m * KernelWidthConv2 * NConv2 + n * NConv2)])== -1){
                    for (size_t bit_counter = 1; bit_counter < NConv2; ++bit_counter) {
                        if(((input[((i + m) * inputWidthConv2 * NConv2 + (j + n) * NConv2)+bit_counter] & kernel[(m * KernelWidthConv2 * NConv2 + n * NConv2)+bit_counter])) == 1)
                        {accumulatedOnes_neg++;}
                    }
                } 
            }
        }
        output[i * OutputWidthConv2 + j] = accumulatedOnes_pos - accumulatedOnes_neg;
    }
}

std::vector<std::vector<float>> ScCudaConv2d(
    const std::vector<float>& polarInputData, 
    const std::vector<float>& polarKernelData, 
    const std::vector<float>& randomMatrix_Input, 
    const std::vector<float>& randomMatrix_Kernel,
    const int bitstream_Length,
    const int height,
    const int width,
    const int heightK){

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Determine the type of the input vector (2D or 3D)
    std::vector<float> inputData = polarInputData;
    std::vector<float> RM_flat = randomMatrix_Input;

    int inputData_size = inputData.size();
    int RM_cols = bitstream_Length;
    int output_size = inputData_size * RM_cols;

    float* d_inputData;
    float* d_RM;
    int8_t* d_output;

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (output_size + blockSize - 1) / blockSize;

    // Allocate input device memory
    cudaCheckError(cudaMalloc(&d_inputData, inputData_size * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_RM, inputData_size * RM_cols * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_output, output_size * sizeof(int8_t)));

    ///SEPARATION KERNEL VARIABLES
    std::vector<float> inputDataK = polarKernelData;
    std::vector<float> RM_flatK = randomMatrix_Kernel;

    int inputData_sizeK = inputDataK.size();
    int RM_colsK = bitstream_Length;
    int output_sizeK = inputData_sizeK * RM_colsK;

    float* d_inputDataK;
    float* d_RMK;
    int8_t* d_outputK;

    // Define grid kernel and block kernel dimensions
    int blockSizeK = 256;
    int numBlocksK = (output_sizeK + blockSizeK - 1) / blockSizeK;      

    // Allocate kernel device memory
    cudaCheckError(cudaMalloc(&d_inputDataK, inputData_sizeK * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_RMK, inputData_sizeK * RM_colsK * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_outputK, output_sizeK * sizeof(int8_t)));

    /////////////////////////////// SC CONV2D KERNEL ///////////////////////////////////////

    const int N = RM_cols; // bitstream length

    int inputHeight = height, inputWidth = width;
    int kernelHeight = heightK, kernelWidth = heightK;
    int outputHeight = inputHeight - kernelHeight + 1;
    int outputWidth = inputWidth - kernelWidth + 1;

    // std::cout << "inputHeight: " << inputHeight << std::endl; 
    // std::cout << "inputWidth: " << inputWidth << std::endl; 
    // std::cout << "kernelHeight: " << kernelHeight << std::endl;
    // std::cout << "outputHeight: " << outputHeight << std::endl; 
    // std::cout << "outputWidth: " << outputWidth << std::endl;

    float* h_output = new float[outputHeight * outputWidth];

    float* d_outputConv2;

                    //time measure
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

    cudaMemcpyToSymbol(inputWidthConv2, &inputWidth, sizeof(int));
    cudaMemcpyToSymbol(KernelHeightConv2, &kernelHeight, sizeof(int));
    cudaMemcpyToSymbol(KernelWidthConv2, &kernelWidth, sizeof(int));
    cudaMemcpyToSymbol(OutputHeightConv2, &outputHeight, sizeof(int));
    cudaMemcpyToSymbol(OutputWidthConv2, &outputWidth, sizeof(int));
    cudaMemcpyToSymbol(NConv2, &N, sizeof(int));

    cudaCheckError(cudaMalloc(&d_outputConv2, outputHeight * outputWidth * sizeof(float)));

    // Define grid and block dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x, (outputHeight + blockDim.y - 1) / blockDim.y);

    // Copy INPUT data to device
    cudaCheckError(cudaMemcpyAsync(d_inputData, inputData.data(), inputData_size * sizeof(float), cudaMemcpyHostToDevice, stream1));
    cudaCheckError(cudaMemcpyAsync(d_RM, RM_flat.data(), inputData_size * RM_cols * sizeof(float), cudaMemcpyHostToDevice, stream1));

    // Copy KERNEL data to device
    cudaCheckError(cudaMemcpyAsync(d_inputDataK, inputDataK.data(), inputData_sizeK * sizeof(float), cudaMemcpyHostToDevice, stream2));
    cudaCheckError(cudaMemcpyAsync(d_RMK, RM_flatK.data(), inputData_sizeK * RM_colsK * sizeof(float), cudaMemcpyHostToDevice, stream2));

    // Launch input_data kernel
    stochasticTensorGenerator<<<numBlocks, blockSize, 0, stream1>>>(d_inputData, d_RM, d_output, inputData_size, RM_cols);
    cudaCheckError(cudaGetLastError());
    
    // Launch kernel_data kernel
    stochasticTensorGenerator<<<numBlocksK, blockSizeK, 0, stream2>>>(d_inputDataK, d_RMK, d_outputK, inputData_sizeK, RM_colsK);
    cudaCheckError(cudaGetLastError());

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Launch the convolution kernel
    conv2D<<<gridDim, blockDim>>>(d_output, d_outputK, d_outputConv2);
    cudaCheckError(cudaGetLastError());

    //cudaDeviceSynchronize();

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_output, d_outputConv2, outputHeight * outputWidth * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<std::vector<float>> outputConv2(outputHeight, std::vector<float>(outputWidth));

                //STOP time measure
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "\n" << std::endl;
            std::cout << "C++ CONV2D End-to-end execution time: " << milliseconds << " ms_" << std::endl;   

    for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            //outputConv2[i][j] = static_cast<double>(h_output[i * outputWidth + j]);
            outputConv2[i][j] = h_output[i * outputWidth + j];
            //std::cout << "outputConv2[i][j] " << outputConv2[i][j] << std::endl;
        }
    }


    // Free device memory
    cudaFree(d_inputData);
    cudaFree(d_RM);
    cudaFree(d_output);

    cudaFree(d_inputDataK);
    cudaFree(d_RMK);
    cudaFree(d_outputK);

    cudaFree(d_outputConv2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Clean up
    delete[] h_output;

    return outputConv2;
}

__constant__ int input_sizeFcL;
__constant__ int output_sizeFcL;
__constant__ int bit_lengthFcL;

///////////////////CUDA FULLY CONNECTED LAYER/////////////////////////
// Kernel function for forward pass of a fully connected layer
__global__ void forward_pass(
    int8_t* input, int8_t* weights, int8_t* biases, float* output) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < output_sizeFcL) {
        float accumulatedOnes_pos= 0;
        float accumulatedOnes_neg = 0;
        int numberOfAccumulations = 0;

        for (size_t i = 0; i < input_sizeFcL; ++i) {
            numberOfAccumulations++;

            if((input[(i * bit_lengthFcL)]) * (weights[((i * output_sizeFcL + j) * bit_lengthFcL)]) == 1){
                for (size_t k = 1; k < bit_lengthFcL; ++k) {
                    if(((input[(i * bit_lengthFcL)+k] & weights[((i * output_sizeFcL + j) * bit_lengthFcL)+k])) == 1)
                        {accumulatedOnes_pos++;}
                }
            } else if ((input[(i * bit_lengthFcL)]) * (weights[((i * output_sizeFcL + j) * bit_lengthFcL)]) == -1){
                for (size_t k = 1; k < bit_lengthFcL; ++k) {
                    if(((input[(i * bit_lengthFcL)+k] & weights[((i * output_sizeFcL + j) * bit_lengthFcL)+k])) == 1)
                        {accumulatedOnes_neg++;}
                }
            }
        }

        if (biases[(j * bit_lengthFcL)] == 1) {
            for (size_t biasIndex = 1; biasIndex < bit_lengthFcL; ++biasIndex) {
                accumulatedOnes_pos += biases[(j * bit_lengthFcL)+biasIndex];
            }
        } else if (biases[(j * bit_lengthFcL)] == -1) {
            for (size_t biasIndex = 1; biasIndex < bit_lengthFcL; ++biasIndex) {
                accumulatedOnes_neg += biases[(j * bit_lengthFcL)+biasIndex];
            }
        }
        numberOfAccumulations++;
        
        output[j] = accumulatedOnes_pos - accumulatedOnes_neg;
    }
}

std::vector<float> ScCudaFcLayer(
    const std::vector<float>& inputs, 
    const std::vector<float>& weights, 
    const std::vector<float>& biases, 
    const std::vector<float>& randomMatrix_input, 
    const std::vector<float>& randomMatrix_weights, 
    const std::vector<float>& randomMatrix_biases, 
    const int num_Outputs,
    const int bitstream_Length){

    // Create CUDA streams
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    
    // Determine the type of the input vector (2D or 3D)
    std::vector<float> RM_flat = randomMatrix_input;

    int inputData_size = inputs.size();
    int RM_cols = bitstream_Length;
    int output_size = inputData_size * RM_cols;

    float* d_inputData;
    float* d_RM;
    int8_t* d_output;

    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (output_size + blockSize - 1) / blockSize;

    // Allocate input device memory
    cudaCheckError(cudaMalloc(&d_inputData, inputData_size * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_RM, inputData_size * RM_cols * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_output, output_size * sizeof(int8_t)));

    ////STG__WEIGHTS__KERNEL//////
    std::vector<float> RM_flatW = randomMatrix_weights;

    std::vector<float> weights_data = weights;
    int inputData_sizeW = weights.size();
    int output_sizeW = inputData_sizeW * RM_cols;

    float* d_inputDataW;
    float* d_RMW;
    int8_t* d_outputW;

    // Define grid kernel and block kernel dimensions
    int blockSizeW = 256;
    int numBlocksW = (output_sizeW + blockSizeW - 1) / blockSizeW;      

    // Allocate kernel device memory
    cudaCheckError(cudaMalloc(&d_inputDataW, inputData_sizeW * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_RMW, inputData_sizeW * RM_cols * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_outputW, output_sizeW * sizeof(int8_t)));

    ////STG__BIASES__KERNEL//////
    std::vector<float> RM_flatB = randomMatrix_biases;

    float* d_inputDataB;
    float* d_RMB;
    int8_t* d_outputB;

    int inputData_sizeB = biases.size();
    int output_sizeB = inputData_sizeB * RM_cols;

    // Define grid kernel and block kernel dimensions
    int blockSizeB = 256;
    int numBlocksB = (output_sizeB + blockSizeB - 1) / blockSizeB;      

    // Allocate kernel device memory
    cudaCheckError(cudaMalloc(&d_inputDataB, inputData_sizeB * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_RMB, inputData_sizeB * RM_cols * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_outputB, output_sizeB * sizeof(int8_t)));

    //int output_sizeFc = biases.size();

    ///////OUTPUT SCFC__KERNEL//////
        // Number of output neurons
    int output_sizeFc =  num_Outputs;
        // Number of input neurons
    const int input_sizeFc = inputs.size(); 

    std::vector<float> h_output(output_sizeFc, 0);
    float* d_outputFc;
    cudaCheckError(cudaMalloc(&d_outputFc, output_sizeFc * sizeof(float)));
    int blockSizeFc = 256;
    int numBlocksFc = (output_sizeFc + blockSizeFc - 1) / blockSizeFc;

                    //time measure
                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);
                    cudaEventRecord(start);

    //CONVERTION INPUT AND KERNEL DATA TO STOCHASTIC
    // Copy INPUT data to device
    cudaCheckError(cudaMemcpyAsync(d_inputData, inputs.data(), inputData_size * sizeof(float), cudaMemcpyHostToDevice, stream1));
    cudaCheckError(cudaMemcpyAsync(d_RM, RM_flat.data(), inputData_size * RM_cols * sizeof(float), cudaMemcpyHostToDevice, stream1));

    // Copy WEIGHTS data to device
    cudaCheckError(cudaMemcpyAsync(d_inputDataW, weights_data.data(), inputData_sizeW * sizeof(float), cudaMemcpyHostToDevice, stream2));
    cudaCheckError(cudaMemcpyAsync(d_RMW, RM_flatW.data(), inputData_sizeW * RM_cols * sizeof(float), cudaMemcpyHostToDevice, stream2));

    // Copy BIAS data to device
    cudaCheckError(cudaMemcpyAsync(d_inputDataB, biases.data(), inputData_sizeB * sizeof(float), cudaMemcpyHostToDevice, stream3));
    cudaCheckError(cudaMemcpyAsync(d_RMB, RM_flatB.data(), inputData_sizeB * RM_cols * sizeof(float), cudaMemcpyHostToDevice, stream3));

    cudaMemcpyToSymbol(input_sizeFcL, &input_sizeFc, sizeof(int));
    cudaMemcpyToSymbol(output_sizeFcL, &output_sizeFc, sizeof(int));
    cudaMemcpyToSymbol(bit_lengthFcL, &RM_cols, sizeof(int));

    ////STG__INPUT__KERNEL//////
    stochasticTensorGenerator<<<numBlocks, blockSize, 0, stream1>>>(d_inputData, d_RM, d_output, inputData_size, RM_cols);
    cudaCheckError(cudaGetLastError());
    
    ////STG__WEIGHTS__KERNEL//////
    stochasticTensorGenerator<<<numBlocksW, blockSizeW, 0, stream2>>>(d_inputDataW, d_RMW, d_outputW, inputData_sizeW, RM_cols);
    cudaCheckError(cudaGetLastError());

    ////STG__BIAS__KERNEL//////
    stochasticTensorGenerator<<<numBlocksB, blockSizeB, 0, stream3>>>(d_inputDataB, d_RMB, d_outputB, inputData_sizeB, RM_cols);
    cudaCheckError(cudaGetLastError());

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    ////__SCFCLAYER__KERNEL//////
    forward_pass<<<numBlocksFc, blockSizeFc>>>(d_output, d_outputW, d_outputB, d_outputFc);
    cudaCheckError(cudaGetLastError());

    //cudaDeviceSynchronize();
    cudaCheckError(cudaMemcpy(h_output.data(), d_outputFc, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

            //STOP time measure
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "\n" << std::endl;
            std::cout << "C++ ScFcLayer End-to-end execution time: " << milliseconds << " ms_" << std::endl;   

    // Free device memory
    cudaFree(d_inputData);
    cudaFree(d_RM);
    cudaFree(d_output);

    cudaFree(d_inputDataW);
    cudaFree(d_RMW);
    cudaFree(d_outputW);

    if (!biases.empty()) {
        cudaFree(d_inputDataB);
        cudaFree(d_RMB);
        cudaFree(d_outputB);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_outputFc);

    return h_output;
}