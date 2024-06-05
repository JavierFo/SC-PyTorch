#ifndef STOCHASTIC_TENSOR_H
#define STOCHASTIC_TENSOR_H

#include <vector>

    // Enum for the bit operation type
    enum BitwiseOperation {
        AND,
        OR,
        XOR,
        NOR,
        XNOR,
        NAND, 
        MUX
    };

    // Enum for the bit stream representation type
    enum BitstreamRepresentation {
        UNIPOLAR,
        BIPOLAR
    };

    // Enum for the random number generator type
    enum RandomNumberGenType {
        LFSR,
        MT19937
    };

class StochasticTensor {
public:
    // 3D vector to store the tensor
    std::vector<std::vector<std::vector<int>>> tensor;

using SizeTuple = std::tuple<size_t, size_t, size_t>;

    // Constructor with random 1s and 0s
    StochasticTensor(int x, int y, int z);

    // Constructor with lfsr based RNG
    StochasticTensor(const std::vector<std::vector<double>>& inputVector, const int bitstreamLength,  RandomNumberGenType type, BitstreamRepresentation mode);

    // Getter for the tensor
    const std::vector<std::vector<std::vector<int>>>& getTensor() const;

    // Method to print the tensor
    void printStochasticTensor() const;

    // Method to get a specific vector at position (i, j) from a random tensor of 1s and 0s
    std::vector<int> getVectorAtFromRandomTensor(int i, int j) const;

    // Method to get a specific vector at position (i, j)
    std::vector<int> getVectorAt(int i, int j) const;

    SizeTuple getSize();

    std::vector<std::vector<double>> toRealTensor(int scale, BitstreamRepresentation mode);

private:
    // Dimensions of the tensor
    int x_dim, y_dim, z_dim;

    // Method to generate the tensor with random 1s and 0s
    void generateRandomTensor();

    // Method to generate the tensor with parameters: input and lfsr RNG array
    // Size of LFSR_basedRandomNumbersArray = Stochastic number bitstream length 
    void generateTensor(const std::vector<std::vector<double>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode);
};

void bipolarStochasticNumberGenerator(const int bitstreamLength, RandomNumberGenType type, int inputRealNumber, std::vector<int>& output,  double minRange, double maxRange);

double normalizeRealNumber(int realNumber, double minRange, double maxRange);

void stochasticNumberGenerator(const int bitstreamLength, RandomNumberGenType type, double inputRealNumber, BitstreamRepresentation mode, std::vector<int>& output);

uint8_t LFSR_StatesGenerator(uint8_t state);

std::vector<int> LFSR_RNG_arrayGenerator(int arrayLength_bitstreamLength, uint8_t lfsr_seed);

uint8_t generateRandomState();

double calculatePx(const std::vector<int>& bstream1, BitstreamRepresentation mode, const std::vector<int>& bstream2 = std::vector<int>());
 
std::vector<int> bitstreamOperation(const std::vector<int>& bitstream1, const std::vector<int>& bitstream2, BitwiseOperation op);

#endif // STOCHASTIC_TENSOR_H
