#ifndef SC_TORCH_H
#define SC_TORCH_H

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
    std::vector<std::vector<std::vector<uint8_t>>> tensor;

    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> scTensor;

    using SizeTuple = std::tuple<size_t, size_t, size_t>;

    using SizeTuple3D = std::tuple<size_t, size_t, size_t, size_t>;

    //default Constructor
    StochasticTensor();

    // Constructor with 2D input
    StochasticTensor(const std::vector<std::vector<double>>& inputVector, const int bitstreamLength,  RandomNumberGenType type, BitstreamRepresentation mode);

    // Constructor with 3D input
    StochasticTensor(const std::vector<std::vector<std::vector<double>>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode);

    // Getter for the tensor
    const std::vector<std::vector<std::vector<uint8_t>>>& getTensor() const;

    // Getter for the 3D tensor
    const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& get3DTensor() const;

    // Method to get a specific vector at position (i, j)
    std::vector<uint8_t> getVectorAt(int i, int j) const;

    std::vector<uint8_t> get3DVectorAt(int i, int j, int k) const;

    SizeTuple getSize();

    SizeTuple3D get3DSize();

    std::vector<std::vector<double>> toRealTensor(int scale, BitstreamRepresentation mode);

    std::vector<std::vector<std::vector<double>>> toReal3DTensor(int scale, BitstreamRepresentation mode);

private:
    // Size of LFSR_basedRandomNumbersArray = Stochastic number bitstream length 
    void generateTensor(const std::vector<std::vector<double>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode);

    void generate3DTensor(const std::vector<std::vector<std::vector<double>>>& inputVector, const int bitstreamLength, RandomNumberGenType type, BitstreamRepresentation mode);
};

void stochasticNumberGenerator(const int bitstreamLength, RandomNumberGenType type, double inputRealNumber, BitstreamRepresentation mode, std::vector<uint8_t>& output);

uint8_t LFSR_StatesGenerator(uint8_t state);

std::vector<int> LFSR_RNG_arrayGenerator(int arrayLength_bitstreamLength, uint8_t lfsr_seed);

double calculatePx(const std::vector<uint8_t>& bstream1, BitstreamRepresentation mode, const std::vector<uint8_t>& bstream2 = std::vector<uint8_t>());
 
std::vector<uint8_t> bitstreamOperation(const std::vector<uint8_t>& bitstream1, const std::vector<uint8_t>& bitstream2, BitwiseOperation op);

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