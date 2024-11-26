#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <bitset>
#include <tuple>
#include <algorithm>
#include <cstdlib>

#include "ScTorch.h"

//#include <torch/extension.h>

// Constructor
StochasticTensor::StochasticTensor() : tensor() {}

StochasticTensor::StochasticTensor(const std::vector<std::vector<double>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode) 
: tensor() {
    generateTensor(inputVector, bitstreamLength, type, mode);
}

StochasticTensor::StochasticTensor(const std::vector<std::vector<std::vector<double>>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode) 
: scTensor() {
    generate3DTensor(inputVector, bitstreamLength, type, mode);
}

// Getter for the tensor
const std::vector<std::vector<std::vector<uint8_t>>>& StochasticTensor::getTensor() const {
    return tensor;
}

// Getter for the tensor
const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& StochasticTensor::get3DTensor() const {
    return scTensor;
}

// Method to generate the tensor with parameters: input and lfsr RNG array
// Size of LFSR_basedRandomNumbersArray = Stochastic number bitstream length 
void StochasticTensor::generateTensor(const std::vector<std::vector<double>>& inputVector, const int bitstreamLength, RandomNumberGenType type,  BitstreamRepresentation mode) {
   std::vector<std::vector<std::vector<uint8_t>>> SCtensor(inputVector.size(), 
        std::vector<std::vector<uint8_t>>(inputVector[0].size(), 
        std::vector<uint8_t>(bitstreamLength)));

        for (size_t i = 0; i < inputVector.size(); ++i) {
            for (size_t j = 0; j < inputVector[i].size(); ++j) {
                stochasticNumberGenerator(bitstreamLength, type,  inputVector[i][j], mode, SCtensor[i][j]);
        }
    }
    tensor = SCtensor;
}

void StochasticTensor::generate3DTensor(const std::vector<std::vector<std::vector<double>>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode){
    using SCTensor3D = std::vector<std::vector<std::vector<std::vector<uint8_t>>>>;
    SCTensor3D scTensor3D_a(inputVector.size());

    for (size_t i = 0; i < inputVector.size(); ++i) {
        scTensor3D_a[i].resize(inputVector[i].size());
        for (size_t j = 0; j < inputVector[i].size(); ++j) {
            scTensor3D_a[i][j].resize(inputVector[i][j].size());
            for (size_t k = 0; k < inputVector[i][j].size(); ++k) {
                stochasticNumberGenerator(bitstreamLength, type,  inputVector[i][j][k], mode, scTensor3D_a[i][j][k]);
            }
        }
    }
    scTensor = scTensor3D_a;
}

std::vector<uint8_t> StochasticTensor::getVectorAt(int i, int j) const {
    if (i >= 0 && i < tensor.size() && j >= 0 && j < tensor[i].size()) {
        return tensor[i][j];
    } else {
        throw std::out_of_range("Index out of range");
    }
}

std::vector<uint8_t> StochasticTensor::get3DVectorAt(int i, int j, int k) const {
    if (i >= 0 && i < scTensor.size() && j >= 0 && j < scTensor[i].size() && k >= 0 && k < scTensor[i][j].size()) {
        return scTensor[i][j][k];
    } else {
        throw std::out_of_range("Index out of range");
    }
}

// Method to get the sizes of each level in a 3D vector
StochasticTensor::SizeTuple StochasticTensor::getSize() {
    size_t depth = tensor.size();
    size_t rows = depth > 0 ? tensor[0].size() : 0;
    size_t columns = (rows > 0 && depth > 0) ? tensor[0][0].size() : 0;

    return std::make_tuple(depth, rows, columns);
}

StochasticTensor::SizeTuple3D StochasticTensor::get3DSize() {
    size_t dimension1 = scTensor.size();
    size_t dimension2 = dimension1 > 0 ? scTensor[0].size() : 0;
    size_t dimension3 = (dimension2 > 0 && dimension1 > 0) ? scTensor[0][0].size() : 0;
    size_t dimension4 = (dimension3 > 0 && dimension2 > 0 && dimension1 > 0) ? scTensor[0][0][0].size() : 0;

    return std::make_tuple(dimension1, dimension2, dimension3, dimension4);
}

// Function to process the input 3D vector and apply the specified operations
std::vector<std::vector<double>> StochasticTensor::toRealTensor(int scale, BitstreamRepresentation mode) {
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
                    rowResult.push_back(((2 * probability) - 1) * scale);
                } else {
                    throw std::invalid_argument("Invalid mode");
                }
        }
        result.push_back(rowResult);
    }
    return result;
}

std::vector<std::vector<std::vector<double>>> StochasticTensor::toReal3DTensor(int scale, BitstreamRepresentation mode) {
    std::vector<std::vector<std::vector<double>>> result;
    
    for (const auto& depth_slice : scTensor) {
        std::vector<std::vector<double>> slice_result;
        for (const auto& row : depth_slice) {
            std::vector<double> rowResult;
            for (const auto& vec : row){
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
                        rowResult.push_back(((2 * probability) - 1) * scale);
                    } else {
                        throw std::invalid_argument("Invalid mode");
                    }
            }
            slice_result.push_back(rowResult);
        }
        result.push_back(slice_result);
    }
    return result;
}

//stochastic into polar number converter
// Method to calculate px and return the appropriate result based on the mode
double calculatePx(const std::vector<uint8_t>& bstream1, BitstreamRepresentation mode, const std::vector<uint8_t>& bstream2) {
    // Combine the vectors if a second vector is provided
    std::vector<uint8_t> combined = bstream1;
    if ((!bstream2.empty()) && (bstream1.size() == bstream2.size())) {
        combined.insert(combined.end(), bstream2.begin(), bstream2.end());
    }

    // Calculate the total number of 1s
    int totalOnes = 0;
    // for (int value : combined) {
    //     if (value == 1) {
    //         ++totalOnes;
    //     }
    // }
    totalOnes = std::count(combined.begin(), combined.end(), 1); //ones counter algorithm O(n) complexity

    // Calculate the length of the combined vector
    int totalLength = combined.size();

    // Calculate px
    double px = static_cast<double>(totalOnes) / totalLength;

    // Return result based on the mode
    if (mode == UNIPOLAR) {
        return px;
    } else if (mode == BIPOLAR) {
        return (2 * px) - 1;
    } else {
        throw std::invalid_argument("Invalid mode");
    }
}

// Function to convert a real number to a stochastic bitstream
void stochasticNumberGenerator(const int bitstreamLength, RandomNumberGenType type, double inputRealNumber, BitstreamRepresentation mode, std::vector<uint8_t>& output) { //,  double minRange, double maxRange
    // Normalize the inputRealNumber to the range [-1, 1]
    // Convert normalized value to probability
    double probability = 0.0;
    if (mode == UNIPOLAR) { probability = inputRealNumber; }
    else if (mode == BIPOLAR) { probability = (inputRealNumber + 1) / 2.0; };

    //std::cout << "probability " << probability << "\n";
    // Resize output vector to the desired bitstream length
    output.resize(bitstreamLength);

    if (type == MT19937) {
        std::random_device rd;  // Non-deterministic random number generator
        std::mt19937 gen(rd()); // Mersenne Twister engine seeded with random_device
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for (int i = 0; i < bitstreamLength; ++i) {
            double randomValue = dis(gen);
            output[i] = (randomValue < probability) ? 1 : 0;
        }
    } else if (type == LFSR) {
        std::random_device rd;
        std::mt19937 gen(rd()); 
        std::uniform_int_distribution<uint8_t> dis(0, 255);
        uint8_t lfsrSeed = dis(gen);
        //randomNumbers = LFSR_RNG_arrayGenerator(bitstreamLength, lfsrSeed);
        for (int i = 0; i < bitstreamLength; ++i) {
            double randomNumber = static_cast<double>(std::bitset<8>(lfsrSeed).to_ulong())/255.0;
            output[i] = (randomNumber < probability) ? 1 : 0;
            lfsrSeed = LFSR_StatesGenerator(lfsrSeed);
        }
    } else {
        throw std::invalid_argument("Invalid type");
    }
}

// Function to generate the next state of the 8 bits LFSR with x8 + x6 + x5 + x4 + 1 polynomial 
uint8_t LFSR_StatesGenerator(uint8_t state) {
    // XOR the bits according to the feedback polynomial
    uint8_t feedback = ((state >> 3) ^ (state >> 4) ^ (state >> 5) ^ (state >> 7)) & 1;
    // Shift right by one bit and set the leftmost bit as the feedback
    state = (state >> 1) | (feedback << 7);
    return state;
}

// Function to perform bitwise operations on two vectors of 1s and 0s
std::vector<uint8_t> bitstreamOperation(const std::vector<uint8_t>& bitstream1, const std::vector<uint8_t>& bitstream2, BitwiseOperation op) {
    std::vector<uint8_t> result(bitstream1.size());
    if (bitstream1.size() != bitstream2.size()) {
        result = bitstream1;
        return result;
    }
    
    //for (size_t i = 0; i < bitstream1.size(); ++i) {
        switch (op) {
            case AND:
                //UNIPOLAR STOCHASTIC MULTIPLIER: transform algorithm O(n) complexity
                //result[i] = bitstream1[i] & bitstream2[i];
                std::transform(bitstream1.begin(), bitstream1.end(), bitstream2.begin(), result.begin(), [](int a, int b) { return a & b; });      
                break;
            case OR:
                //result[i] = bitstream1[i] | bitstream2[i];
                std::transform(bitstream1.begin(), bitstream1.end(), bitstream2.begin(), result.begin(), [](int a, int b) { return a | b; });      
                break;
            case XOR:
                //result[i] = bitstream1[i] ^ bitstream2[i];
                std::transform(bitstream1.begin(), bitstream1.end(), bitstream2.begin(), result.begin(), [](int a, int b) { return a ^ b; });      
                break;
            case NOR:
                //result[i] = ~(bitstream1[i] | bitstream2[i]) & 1; // & 1 to ensure the result is either 0 or 1
                std::transform(bitstream1.begin(), bitstream1.end(), bitstream2.begin(), result.begin(), [](int a, int b) { return !(a | b); });      
                break;
            case XNOR:
                //BIPOLAR STOCHASTIC MULTIPLIER: transform algorithm O(n) complexity
                //result[i] = ~(bitstream1[i] ^ bitstream2[i]) & 1; // & 1 to ensure the result is either 0 or 1
                std::transform(bitstream1.begin(), bitstream1.end(), bitstream2.begin(), result.begin(), [](int a, int b) { return !(a ^ b); });      
                break;
            case NAND:
                //result[i] = ~(bitstream1[i] & bitstream2[i]) & 1; // & 1 to ensure the result is either 0 or 1
                std::transform(bitstream1.begin(), bitstream1.end(), bitstream2.begin(), result.begin(), [](int a, int b) { return !(a & b); });      
                break;
            case MUX: 
                //random_value = dis(gen);
                //result[i] = (random_value < 0.5) ? bitstream1[i] : bitstream2[i];
                break;
            default:
                throw std::invalid_argument("Invalid operation");
        }
    //}
    return result;
}

std::vector<uint8_t> concatenateSCVectors(const std::vector<uint8_t>& vector1, const std::vector<uint8_t>& vector2) {
    if (vector2.size() > 1 && std::any_of(vector2.begin(), vector2.end(), [](uint8_t i) { return i == 1; })) {
        std::vector<uint8_t> result = vector1;
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
        std::vector<uint8_t> addedScOutput_acc(1, 0);
      for (int kh = 0; kh < kernel_height; kh += dilation) {
        for (int kw = 0; kw < kernel_width; kw += dilation) {
          // Handle padding with boundary checks
          int in_h = oh * stride - padding + kh;
          int in_w = ow * stride - padding + kw;

          if (in_h >= 0 && in_h < in_height && 
              in_w >= 0 && in_w < in_width) {
            std::vector<uint8_t> scMultiplication = bitstreamOperation(input.getVectorAt(in_h, in_w), kernel.getVectorAt(kh,kw), XNOR);
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
        std::vector<uint8_t> addedScOutput_acc(1, 0);
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
                std::vector<uint8_t> scMultiplication = bitstreamOperation(input.get3DVectorAt(in_d, in_h, in_w), kernel.get3DVectorAt(kd, kh, kw), XNOR);
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
        std::vector<uint8_t> accumulatedMultiplication(1, 0);
        std::vector<uint8_t> accumulatedBias(1, 0);
        for (int i = 0; i < input_size; ++i) {
          std::vector<uint8_t> scMultiplication = bitstreamOperation(scInput.getVectorAt(0,i),scWeight.getVectorAt(i,j), XNOR);
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

// D11_MODULE(sc_torch_cpp, m) {  // Ensure the module name matches
//     m.def("ScConv2d", &ScConv2d, "A stochastic 2D convolution function");
//     m.def("ScConv3d", &ScConv3d, "A stochastic 3D convolution function");
//     m.def("forward", &forward, "A stochastic fully connected layer function");
//     m.def("sigmoid", &sigmoid, "Sigmoid Activation function");
//     m.def("relu", &relu, "relu Activation function");
//     m.def("leaky_relu", &leaky_relu, "leaky_relu Activation function");
//     m.def("softmax", &softmax, "softmax Activation function");
//     m.def("tanh", &tanh, "tanh Activation function");

//     // Expose the ScFcLayer class
//     pybind11::class_<ScFcLayer>(m, "ScFcLayer")
//         .def(pybind11::init<const std::vector<std::vector<double>>&, const std::vector<double>&, int,  RandomNumberGenType, BitstreamRepresentation>())
//         .def(pybind11::init<int, int, int,  RandomNumberGenType, BitstreamRepresentation>())
// }

// PYBIND11_MODULE(stochastic_tensor_cpp, m) {  // Ensure the module name matches
//      pybind11::enum_<RandomNumberGenType>(m, "RandomNumberGenType")
//         .value("LFSR", RandomNumberGenType::LFSR)
//         .value("MT19937", RandomNumberGenType::MT19937)
//         .export_values();

//     pybind11::enum_<BitwiseOperation>(m, "BitwiseOperation")
//         .value("AND", BitwiseOperation::AND)
//         .value("OR", BitwiseOperation::OR)
//         .value("XOR", BitwiseOperation::XOR)
//         .value("NOR", BitwiseOperation::NOR)
//         .value("XNOR", BitwiseOperation::XNOR)
//         .value("NAND", BitwiseOperation::NAND)
//         .value("MUX", BitwiseOperation::MUX)
//         .export_values();

//     pybind11::enum_<BitstreamRepresentation>(m, "BitstreamRepresentation")
//         .value("UNIPOLAR", BitstreamRepresentation::UNIPOLAR)
//         .value("BIPOLAR", BitstreamRepresentation::BIPOLAR)
//         .export_values();

//     // Expose the StochasticTensor class
//     pybind11::class_<StochasticTensor>(m, "StochasticTensor")
//         .def(pybind11::init<const std::vector<std::vector<double>>&, int, RandomNumberGenType, BitstreamRepresentation>())
//         .def(pybind11::init<const std::vector<std::vector<std::vector<double>>>&, int, RandomNumberGenType, BitstreamRepresentation>())
// }