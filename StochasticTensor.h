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

    std::vector<std::vector<std::vector<std::vector<int>>>> scTensor;

    using SizeTuple = std::tuple<size_t, size_t, size_t>;

    using SizeTuple3D = std::tuple<size_t, size_t, size_t, size_t>;

    //default Constructor
    StochasticTensor();

    // Constructor with 2D input
    StochasticTensor(const std::vector<std::vector<double>>& inputVector, const int bitstreamLength,  RandomNumberGenType type, BitstreamRepresentation mode);

    // Constructor with 3D input
    StochasticTensor(const std::vector<std::vector<std::vector<double>>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode);

    // Getter for the tensor
    const std::vector<std::vector<std::vector<int>>>& getTensor() const;

    // Getter for the 3D tensor
    const std::vector<std::vector<std::vector<std::vector<int>>>>& get3DTensor() const;

    // Method to print the tensor
    void printStochasticTensor() const;

    // Method to get a specific vector at position (i, j)
    std::vector<int> getVectorAt(int i, int j) const;

    std::vector<int> get3DVectorAt(int i, int j, int k) const;

    SizeTuple getSize();

    SizeTuple3D get3DSize();

    std::vector<std::vector<double>> toRealTensor(int scale, BitstreamRepresentation mode);

    std::vector<std::vector<std::vector<double>>> toReal3DTensor(int scale, BitstreamRepresentation mode);

private:
    // Size of LFSR_basedRandomNumbersArray = Stochastic number bitstream length 
    void generateTensor(const std::vector<std::vector<double>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode);

    void generate3DTensor(const std::vector<std::vector<std::vector<double>>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode);
};

void bipolarStochasticNumberGenerator(const int bitstreamLength, RandomNumberGenType type, int inputRealNumber, std::vector<int>& output,  double minRange, double maxRange);

double normalizeRealNumber(int realNumber, double minRange, double maxRange);

void stochasticNumberGenerator(const int bitstreamLength, RandomNumberGenType type, double inputRealNumber, BitstreamRepresentation mode, std::vector<int>& output);

uint8_t LFSR_StatesGenerator(uint8_t state);

std::vector<int> LFSR_RNG_arrayGenerator(int arrayLength_bitstreamLength, uint8_t lfsr_seed);

uint8_t generateRandomState();

double calculatePx(const std::vector<int>& bstream1, BitstreamRepresentation mode, const std::vector<int>& bstream2 = std::vector<int>());
 
std::vector<int> bitstreamOperation(const std::vector<int>& bitstream1, const std::vector<int>& bitstream2, BitwiseOperation op);

std::vector<std::vector<double>> SC_2Dconv (const StochasticTensor input, const StochasticTensor kernel);

std::vector<int> concatenateSCVectors(const std::vector<int>& vector1, const std::vector<int>& vector2);

#endif // STOCHASTIC_TENSOR_H
