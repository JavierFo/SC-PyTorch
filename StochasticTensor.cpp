#include "StochasticTensor.h"
#include <iostream>
#include <stdexcept>
#include <random>

// Constructor
StochasticTensor::StochasticTensor(int x, int y, int z)
    : x_dim(x), y_dim(y), z_dim(z), tensor(x, std::vector<std::vector<int>>(y, std::vector<int>(z))) {
    generateRandomTensor();
}

StochasticTensor::StochasticTensor(const std::vector<std::vector<double>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode) 
: tensor() {
    generateTensor(inputVector, bitstreamLength, type, mode);
}

// Getter for the tensor
const std::vector<std::vector<std::vector<int>>>& StochasticTensor::getTensor() const {
    return tensor;
}

// Method to print the tensor
void StochasticTensor::printStochasticTensor() const {
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
void StochasticTensor::generateTensor(const std::vector<std::vector<double>>& inputVector, const int bitstreamLength, RandomNumberGenType type,  BitstreamRepresentation mode) {
   std::vector<std::vector<std::vector<int>>> SCtensor(inputVector.size());

    for (size_t i = 0; i < inputVector.size(); ++i) {
        SCtensor[i].resize(inputVector[i].size());
        for (size_t j = 0; j < inputVector[i].size(); ++j) {
            stochasticNumberGenerator(bitstreamLength, type,  inputVector[i][j], mode, SCtensor[i][j]);
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

// Method to calculate px and return the appropriate result based on the mode
double calculatePx(const std::vector<int>& bstream1, BitstreamRepresentation mode, const std::vector<int>& bstream2) {
    // Combine the vectors if a second vector is provided
    std::vector<int> combined = bstream1;
    if ((!bstream2.empty()) && (bstream1.size() == bstream2.size())) {
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
        return (2 * px) - 1;
    } else {
        throw std::invalid_argument("Invalid mode");
    }
}

// Function to normalize the real number to the range [-1, 1]
double normalizeRealNumber(int realNumber, double minRange, double maxRange) {
    //return 2 * ((static_cast<double>(realNumber) - minRange) / (maxRange - minRange)) - 1;
    return realNumber/maxRange;
}

// Function to convert a real number to a stochastic bitstream
void stochasticNumberGenerator(const int bitstreamLength, RandomNumberGenType type, double inputRealNumber, BitstreamRepresentation mode, std::vector<int>& output) { //,  double minRange, double maxRange
    // Normalize the inputRealNumber to the range [-1, 1]
    //double normalizedValue = normalizeRealNumber(inputRealNumber, minRange, maxRange);
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
std::vector<int> bitstreamOperation(const std::vector<int>& bitstream1, const std::vector<int>& bitstream2, BitwiseOperation op) {
    std::vector<int> result(bitstream1.size());
    if (bitstream1.size() != bitstream2.size()) {
        result = bitstream1;
        return result;
    }
    
    //RNG for the MUX
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> dis(0.0, 1.0); // Define the range
    double random_value = dis(gen);

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
            case MUX: 
                random_value = dis(gen);
                //result[i] = (select[i] == 1) ? bitstream1[i] : bitstream2[i];
                result[i] = (random_value < 0.5) ? bitstream1[i] : bitstream2[i];
                // if (random_value < 0.5) {
                //     result[i] = bitstream1[i];
                // } else {
                //     result[i] = bitstream2[i];
                // }
                break;
            default:
                throw std::invalid_argument("Invalid operation");
        }
    }
    return result;
}

// // Function to generate an array of lfsr based random numbers
// std::vector<int> LFSR_RNG_arrayGenerator(int arrayLength_bitstreamLength, uint8_t lfsr_seed) {
//     std::vector<int> randomNumbersArray(arrayLength_bitstreamLength);
//     for (int i = 0; i < arrayLength_bitstreamLength; ++i) {
//         randomNumbersArray[i] = std::bitset<8>(lfsr_seed).to_ulong(); 
//         lfsr_seed = LFSR_StatesGenerator(lfsr_seed);
//     }
//     return randomNumbersArray;
// }

// Function to create stochastic number bitstreams
// void stochasticNumberGenerator(const int bitstreamLength, RandomNumberGenType type, int inputRealNumber, std::vector<int> &output) {

//     std::vector<int> randomNumbers(bitstreamLength);
//     output.resize(bitstreamLength);
//     if (type == MT19937) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_int_distribution<uint8_t> dis(0, 255);
//         for (int i = 0; i < bitstreamLength; ++i) {
//             double randomValue = dis(gen);
//             output[i] = (randomValue < inputRealNumber) ? 1 : 0;
//         }
//     } else if (type == LFSR) {
//         uint8_t lfsrSeed = generateRandomState();
//         randomNumbers = LFSR_RNG_arrayGenerator(bitstreamLength, lfsrSeed);
//         for (int i = 0; i < bitstreamLength; ++i) {
//         if (randomNumbers[i] < inputRealNumber) {
//             output[i] = 1;
//         } else {
//             output[i] = 0;
//         }
//     }
//     } else {
//         throw std::invalid_argument("Invalid type");
//     }
// }