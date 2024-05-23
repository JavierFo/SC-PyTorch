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
        NAND
    };

    // Enum for the bit stream representation type
    enum bitstreamRepresentation {
        UNIPOLAR,
        BIPOLAR
    };

class StochasticTensor {
public:

using SizeTuple = std::tuple<size_t, size_t, size_t>;

    // Constructor with random 1s and 0s
    StochasticTensor(int x, int y, int z);

    // Constructor with lfsr based RNG
    StochasticTensor(const std::vector<std::vector<int>>& inputVector, const std::vector<int>&LFSR_basedRandomNumbersArray);

    // Getter for the tensor
    const std::vector<std::vector<std::vector<int>>>& getTensor() const;

    // Method to print the tensor
    void printTensor() const;

    // Method to get a specific vector at position (i, j) from a random tensor of 1s and 0s
    std::vector<int> getVectorAtFromRandomTensor(int i, int j) const;

    // Method to get a specific vector at position (i, j)
    std::vector<int> getVectorAt(int i, int j) const;

    SizeTuple getSize();

private:
    // Dimensions of the tensor
    int x_dim, y_dim, z_dim;

    // 3D vector to store the tensor
    std::vector<std::vector<std::vector<int>>> tensor;

    // Method to generate the tensor with random 1s and 0s
    void generateRandomTensor();

    // Method to generate the tensor with parameters: input and lfsr RNG array
    // Size of LFSR_basedRandomNumbersArray = Stochastic number bitstream length 
    void generateTensor(const std::vector<std::vector<int>>& inputVector, const std::vector<int>&LFSR_basedRandomNumbersArray);
};

uint8_t LFSR_StatesGenerator(uint8_t state);

void stochasticNumberGenerator(const std::vector<int>&randomNumbers, int inputRealNumber, std::vector<int> &output);

std::vector<int> LFSR_RNG_arrayGenerator(int arrayLength_bitstreamLength, uint8_t lfsr_seed);


double calculatePx(const std::vector<int>& bstream1, bitstreamRepresentation mode, const std::vector<int>& bstream2 = std::vector<int>());
 
std::vector<int> bitstreamOperation(const std::vector<int>& bitstream1, const std::vector<int>& bitstream2, BitwiseOperation op);

#endif // STOCHASTIC_TENSOR_H
