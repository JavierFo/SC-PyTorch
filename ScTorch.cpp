#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include "ScTorch.h"
#include "StochasticTensor.h"


std::vector<int> concatenateSCVectors(const std::vector<int>& vector1, const std::vector<int>& vector2) {
    if (vector2.size() > 1 && std::any_of(vector2.begin(), vector2.end(), [](int i) { return i == 1; })) {
        std::vector<int> result = vector1;
        result.insert(result.end(), vector2.begin(), vector2.end());
        return result;
    }
    return vector1;
}

///////////////////////////////////////////////////////////////////////////
// Function to perform stochastic 2D convolution
std::vector<std::vector<double>> ScConv2d(
    const StochasticTensor input, const StochasticTensor kernel,
    int padding, int stride, int dilation) {

    StochasticTensor SCtensorInput = input;
    StochasticTensor SCtensorKernel = kernel;

    StochasticTensor::SizeTuple inputSizes = SCtensorInput.getSize();
    StochasticTensor::SizeTuple kernelSizes = SCtensorKernel.getSize();

  // Validate input and kernel dimensions
  int in_height = std::get<0>(inputSizes);
  int in_width = std::get<1>(inputSizes);

  int kernel_height = std::get<0>(kernelSizes);
  int kernel_width = std::get<1>(kernelSizes);

  if (kernel_height % dilation != 0 || kernel_width % dilation != 0) {
    throw std::invalid_argument("Incompatible dilation and kernel dimensions");
  }

  // Calculate output dimensions with padding
  int out_height = (in_height - kernel_height + 2 * padding) / stride + 1;
  int out_width = (in_width - kernel_width + 2 * padding) / stride + 1;

  const double inputBitstreamSize = std::get<2>(inputSizes);

  // Initialize output with zeros
  std::vector<std::vector<double>> stochasticOutputWthAcc(out_height, std::vector<double>(out_width, 0));

  // Perform convolution
  for (int oh = 0; oh < out_height; ++oh) {
    for (int ow = 0; ow < out_width; ++ow) {
        std::vector<int> addedScOutput_acc(1, 0);
      for (int kh = 0; kh < kernel_height; kh += dilation) {
        for (int kw = 0; kw < kernel_width; kw += dilation) {
          // Handle padding with boundary checks
          int in_h = oh * stride - padding + kh;
          int in_w = ow * stride - padding + kw;

          if (in_h >= 0 && in_h < in_height && 
              in_w >= 0 && in_w < in_width) {
            std::vector<int> scMultiplication = bitstreamOperation(input.getVectorAt(in_h, in_w), kernel.getVectorAt(kh,kw), XNOR);
            addedScOutput_acc = concatenateSCVectors(scMultiplication,addedScOutput_acc);
            //output[oh][ow] += input[in_h][in_w] * kernel[kh][kw];
          }
        }
      }
    double scale = addedScOutput_acc.size()/inputBitstreamSize;
    stochasticOutputWthAcc[oh][ow] = calculatePx(addedScOutput_acc, BIPOLAR)*scale;
    }
  }

  return stochasticOutputWthAcc;
}

///////////////////////////////////////////////////////////////////////////
// Function to perform stochastic 3D convolution
std::vector<std::vector<std::vector<double>>> ScConv3d(
    const StochasticTensor input, const StochasticTensor kernel,
    int padding, int stride, int dilation) {

    StochasticTensor SCtensorInput = input;
    StochasticTensor SCtensorKernel = kernel;

    StochasticTensor::SizeTuple3D inputSizes = SCtensorInput.get3DSize();
    StochasticTensor::SizeTuple3D kernelSizes = SCtensorKernel.get3DSize();

  int in_depth = std::get<0>(inputSizes);
  int in_height = std::get<1>(inputSizes);
  int in_width = std::get<2>(inputSizes);

  int kernel_depth = std::get<0>(kernelSizes);
  int kernel_height = std::get<1>(kernelSizes);
  int kernel_width = std::get<2>(kernelSizes);

  if (in_depth != kernel_depth || kernel_depth % dilation != 0) {
    throw std::invalid_argument("Incompatible input and kernel depth dimensions");
  }

  // Calculate output dimensions with padding
  int out_depth = (in_depth - kernel_depth + 2 * padding) / stride + 1;
  int out_height = (in_height - kernel_height + 2 * padding) / stride + 1;
  int out_width = (in_width - kernel_width + 2 * padding) / stride + 1;

  const double inputBitstreamSize = std::get<3>(inputSizes);

  // Initialize output with zeros
  std::vector<std::vector<std::vector<double>>> output(out_depth, 
      std::vector<std::vector<double>>(out_height, std::vector<double>(out_width, 0)));

  // Perform convolution
  for (int od = 0; od < out_depth; ++od) {
    for (int oh = 0; oh < out_height; ++oh) {
      for (int ow = 0; ow < out_width; ++ow) {
        std::vector<int> addedScOutput_acc(1, 0);
        for (int kd = 0; kd < kernel_depth; kd += dilation) {
          for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
              // Handle padding with boundary checks
              int in_d = od * stride - padding + kd;
              int in_h = oh * stride - padding + kh;
              int in_w = ow * stride - padding + kw;

              if (in_d >= 0 && in_d < in_depth && 
                  in_h >= 0 && in_h < in_height && 
                  in_w >= 0 && in_w < in_width) {
                std::vector<int> scMultiplication = bitstreamOperation(input.get3DVectorAt(in_d, in_h, in_w), kernel.get3DVectorAt(kd, kh, kw), XNOR);
                addedScOutput_acc = concatenateSCVectors(scMultiplication,addedScOutput_acc);
                //output[od][oh][ow] += input[in_d][in_h][in_w] * kernel[kd][kh][kw];
              }
            }
          }
        }
        double scale = addedScOutput_acc.size()/inputBitstreamSize;
        output[od][oh][ow] = calculatePx(addedScOutput_acc, BIPOLAR)*scale;
      }
    }
  }

  return output;
}

///////////////////////////////////////////////////////////////////////////    
// Constructor with specified weights and bias
ScFcLayer::ScFcLayer(const std::vector<std::vector<double>>& weights, const std::vector<double>& bias, 
  const int bitstreamLength,  RandomNumberGenType type, BitstreamRepresentation mode)
    : weights(weights), bias(bias), input_size(weights.size()), output_size(bias.size()),
      scWeight(StochasticTensor(weights, bitstreamLength, type, mode)),
      scBias(StochasticTensor({bias}, bitstreamLength, type, mode)),
      bitstreamLength(bitstreamLength), type(type), mode(mode) {}

// Constructor with random weights and bias
ScFcLayer::ScFcLayer(int input_size, int output_size, 
  const int bitstreamLength,  RandomNumberGenType type, BitstreamRepresentation mode)
    : input_size(input_size), output_size(output_size),
      bitstreamLength(bitstreamLength), type(type), mode(mode) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    weights.resize(input_size, std::vector<double>(output_size));
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            weights[i][j] = dis(gen);
        }
    }

    bias.resize(output_size);
    for (int i = 0; i < output_size; ++i) {
        bias[i] = dis(gen);
    }

    scWeight = StochasticTensor(weights, bitstreamLength, type, mode);
    scBias = StochasticTensor({bias}, bitstreamLength, type, mode);
}

// Forward pass without activation function
std::vector<double> ScFcLayer::forward(const std::vector<double>& inputs) {
    if (inputs.size() != input_size) {
        throw std::invalid_argument("Input size does not match the layer's input size.");
    }
    StochasticTensor scInput = StochasticTensor({inputs}, bitstreamLength, type, mode);
    std::vector<double> outputs(output_size, 0.0);

    //std::vector<double> outputTEMPORAL(output_size, 0.0);

    for (int j = 0; j < output_size; ++j) {
        std::vector<int> accumulatedMultiplication(1, 0);
        std::vector<int> accumulatedBias(1, 0);
        for (int i = 0; i < input_size; ++i) {
          std::vector<int> scMultiplication = bitstreamOperation(scInput.getVectorAt(0,i),scWeight.getVectorAt(i,j), XNOR);
          accumulatedMultiplication = concatenateSCVectors(scMultiplication,accumulatedMultiplication);
          //outputTEMPORAL[j] += inputs[i] * weights[i][j];
        }
        accumulatedBias = concatenateSCVectors(scBias.getVectorAt(0,j),accumulatedMultiplication);
        double scale = double(accumulatedBias.size())/double(bitstreamLength);
        outputs[j] = calculatePx(accumulatedBias, mode)*scale;
        //outputTEMPORAL[j] += bias[j];
        //std::cout << outputTEMPORAL[j] << " ";
    }
    return outputs;
}

// Sigmoid activation function
std::vector<double> ScFcLayer::sigmoid(const std::vector<double>& inputs) {
    std::vector<double> outputs = forward(inputs);
    for (double& output : outputs) {
        output = 1 / (1 + std::exp(-output));
    }
    return outputs;
}

// ReLU activation function
std::vector<double> ScFcLayer::relu(const std::vector<double>& inputs) {
    std::vector<double> outputs = forward(inputs);
    for (double& output : outputs) {
        output = std::max(0.0, output);
    }
    return outputs;
}

// Leaky ReLU activation function
std::vector<double> ScFcLayer::leaky_relu(const std::vector<double>& inputs, double alpha) {
    std::vector<double> outputs = forward(inputs);
    for (double& output : outputs) {
        output = (output > 0) ? output : alpha * output;
    }
    return outputs;
}

// Tanh activation function
std::vector<double> ScFcLayer::tanh(const std::vector<double>& inputs) {
    std::vector<double> outputs = forward(inputs);
    for (double& output : outputs) {
        output = std::tanh(output);
    }
    return outputs;
}

// Softmax activation function
std::vector<double> ScFcLayer::softmax(const std::vector<double>& inputs) {
    std::vector<double> outputs = forward(inputs);
    double max_output = *std::max_element(outputs.begin(), outputs.end());
    double sum_exp = 0.0;
    for (double& output : outputs) {
        output = std::exp(output - max_output);
        sum_exp += output;
    }
    for (double& output : outputs) {
        output /= sum_exp;
    }
    return outputs;
}
