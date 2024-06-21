#ifndef SC_TORCH_H
#define SC_TORCH_H

#include <vector>
#include "StochasticTensor.h"

std::vector<std::vector<double>> ScConv2d(
    const StochasticTensor input, const StochasticTensor kernel,
    int padding, int stride, int dilation);

std::vector<std::vector<std::vector<double>>> ScConv3d(
    const StochasticTensor input, const StochasticTensor kernel,
    int padding, int stride, int dilation);

class ScFcLayer {
 private:
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    int input_size;
    int output_size;
    StochasticTensor scWeight;
    StochasticTensor scBias;
    int bitstreamLength;  
    RandomNumberGenType type; 
    BitstreamRepresentation mode;

public: 

    ScFcLayer(const std::vector<std::vector<double>>& weights, const std::vector<double>& bias, 
      const int bitstreamLength,  RandomNumberGenType type, BitstreamRepresentation mode);

    ScFcLayer(int input_size, int output_size, 
      const int bitstreamLength,  RandomNumberGenType type, BitstreamRepresentation mode);

    std::vector<double> forward(const std::vector<double>& inputs);

    std::vector<double> sigmoid(const std::vector<double>& inputs);

    std::vector<double> relu(const std::vector<double>& inputs);

    std::vector<double> leaky_relu(const std::vector<double>& inputs, double alpha = 0.01);

    std::vector<double> tanh(const std::vector<double>& inputs);

    std::vector<double> softmax(const std::vector<double>& inputs);

};
    
#endif // SC_TORCH_H
