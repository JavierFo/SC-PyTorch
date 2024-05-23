#include "StochasticTensor.h"
#include <iostream>
#include <stdexcept>
#include <random>

// Constructor
StochasticTensor::StochasticTensor(int x, int y, int z)
    : x_dim(x), y_dim(y), z_dim(z), tensor(x, std::vector<std::vector<int>>(y, std::vector<int>(z))) {
    generateRandomTensor();
}

StochasticTensor::StochasticTensor(const std::vector<std::vector<int>>& inputVector, const std::vector<int>&LFSR_basedRandomNumbersArray) 
: tensor() {
    generateTensor(inputVector, LFSR_basedRandomNumbersArray);
}

// Getter for the tensor
const std::vector<std::vector<std::vector<int>>>& StochasticTensor::getTensor() const {
    return tensor;
}

// Method to print the tensor
void StochasticTensor::printTensor() const {
    std::cout << "{" << std::endl;
    for (const auto& matrix : tensor) {
        std::cout << "  {" << std::endl;
        for (const auto& row : matrix) {
            std::cout << "    { ";
            for (int value : row) {
                std::cout << value << " ";
            }
            std::cout << "}" << std::endl;
        }
        std::cout << "  }" << std::endl;
    }
    std::cout << "}" << std::endl;
}

// Method to generate the tensor with parameters: input and lfsr RNG array
// Size of LFSR_basedRandomNumbersArray = Stochastic number bitstream length 
void StochasticTensor::generateTensor(const std::vector<std::vector<int>>& inputVector, const std::vector<int>&LFSR_basedRandomNumbersArray) {
   std::vector<std::vector<std::vector<int>>> SCtensor(inputVector.size());

    for (size_t i = 0; i < inputVector.size(); ++i) {
        SCtensor[i].resize(inputVector[i].size());
        for (size_t j = 0; j < inputVector[i].size(); ++j) {
            stochasticNumberGenerator(LFSR_basedRandomNumbersArray, inputVector[i][j], SCtensor[i][j]);
        }
    }

    tensor = SCtensor;
}

std::vector<int> StochasticTensor::getVectorAt(int i, int j) const {
    if (i >= 0 && i < tensor.size() && j >= 0 && j < tensor[i].size()) {
        return tensor[i][j];
    } else {
        throw std::out_of_range("Index out of range");
    }
}

// Method to generate the tensor with random 1s and 0s
void StochasticTensor::generateRandomTensor() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < x_dim; ++i) {
        for (int j = 0; j < y_dim; ++j) {
            for (int k = 0; k < z_dim; ++k) {
                tensor[i][j][k] = dis(gen);
            }
        }
    }
}

// Method to get a specific vector at position (i, j)
std::vector<int> StochasticTensor::getVectorAtFromRandomTensor(int i, int j) const {
    if (i >= 0 && i < x_dim && j >= 0 && j < y_dim) {
        return tensor[i][j];
    } else {
        throw std::out_of_range("Index out of bounds");
    }
}

// Method to get the sizes of each level in a 3D vector
StochasticTensor::SizeTuple StochasticTensor::getSize() {
    size_t depth = tensor.size();
    size_t rows = depth > 0 ? tensor[0].size() : 0;
    size_t columns = (rows > 0 && depth > 0) ? tensor[0][0].size() : 0;

    return std::make_tuple(depth, rows, columns);
}

// Method to calculate px and return the appropriate result based on the mode
double calculatePx(const std::vector<int>& bstream1, bitstreamRepresentation mode, const std::vector<int>& bstream2) {
    // Combine the vectors if a second vector is provided
    std::vector<int> combined = bstream1;
    if (!bstream2.empty()) {
        combined.insert(combined.end(), bstream2.begin(), bstream2.end());
    }

    // Calculate the total number of 1s
    int totalOnes = 0;
    for (int value : combined) {
        if (value == 1) {
            ++totalOnes;
        }
    }

    // Calculate the length of the combined vector
    int totalLength = combined.size();

    // Calculate px
    double px = static_cast<double>(totalOnes) / totalLength;

    // Return result based on the mode
    if (mode == UNIPOLAR) {
        return px;
    } else if (mode == BIPOLAR) {
        return 2 * px - 1;
    } else {
        throw std::invalid_argument("Invalid mode");
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

void stochasticNumberGenerator(const std::vector<int>&randomNumbers, int inputRealNumber, std::vector<int> &output) {

    // Resize the output vector to ensure it has enough space
    int bitstreamLength =randomNumbers.size();
    output.resize(bitstreamLength);
    
    for (int i = 0; i < bitstreamLength; ++i) {
        if (randomNumbers[i] < inputRealNumber) {
            output[i] = 1;
        } else {
            output[i] = 0;
        }
    }
}

// Function to generate an array of lfsr based random numbers
std::vector<int> LFSR_RNG_arrayGenerator(int arrayLength_bitstreamLength, uint8_t lfsr_seed) {
    std::vector<int> randomNumbersArray(arrayLength_bitstreamLength);
    for (int i = 0; i < arrayLength_bitstreamLength; ++i) {
        randomNumbersArray[i] = std::bitset<8>(lfsr_seed).to_ulong(); 
        lfsr_seed = LFSR_StatesGenerator(lfsr_seed);
    }
    return randomNumbersArray;
}

// Function to perform bitwise operations on two vectors of 1s and 0s
std::vector<int> bitstreamOperation(const std::vector<int>& bitstream1, const std::vector<int>& bitstream2, BitwiseOperation op) {
    if (bitstream1.size() != bitstream2.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }

    std::vector<int> result(bitstream1.size());

    for (size_t i = 0; i < bitstream1.size(); ++i) {
        switch (op) {
            case AND:
                result[i] = bitstream1[i] & bitstream2[i];
                break;
            case OR:
                result[i] = bitstream1[i] | bitstream2[i];
                break;
            case XOR:
                result[i] = bitstream1[i] ^ bitstream2[i];
                break;
            case NOR:
                result[i] = ~(bitstream1[i] | bitstream2[i]) & 1; // & 1 to ensure the result is either 0 or 1
                break;
            case XNOR:
                result[i] = ~(bitstream1[i] ^ bitstream2[i]) & 1; // & 1 to ensure the result is either 0 or 1
                break;
            case NAND:
                result[i] = ~(bitstream1[i] & bitstream2[i]) & 1; // & 1 to ensure the result is either 0 or 1
                break;
            default:
                throw std::invalid_argument("Invalid operation");
        }
    }
    return result;
}
