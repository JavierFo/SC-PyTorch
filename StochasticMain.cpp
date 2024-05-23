#include "StochasticTensor.h"
#include <iostream>
#include <vector>

/*int main() {
    //int x = 3, y = 3, z = 5; // Dimensions of the tensor
    //StochasticTensor st(x, y, z);

        // Example kernel matrix
    std::vector<std::vector<int>> kernel = {
        {0, 185, 276},
        {213, 0, 33},
        {44, 255, 0}
    };

    int N = 3; // size of lfsr based random numbers / bitstream length
    uint8_t lfsr_state = 0b11001100; // initial 8bits lfsr state : Period (2^{n}-1) = max 255, next x9 + x5 + 1: max 511
     
    //random numbers array for SNG 
    std::vector<int> randomNumbers = LFSR_RNG_arrayGenerator(N, lfsr_state);

    for (int i = 0; i < N; ++i) {
        std::cout << randomNumbers[i] << " ";
    }

    StochasticTensor SCtensor(kernel, randomNumbers);
    // Print the tensor
    SCtensor.printTensor();
    StochasticTensor::SizeTuple sizes = SCtensor.getSize();

    try {
        // Extract the vector at position (1,1)
        std::vector<int> extractedVector0 = SCtensor.getVectorAt(0, 1);
        std::vector<int> extractedVector1 = SCtensor.getVectorAt(1, 0);
        std::vector<int> extractedVector2 = SCtensor.getVectorAt(2, 1);

           double o = calculatePx(extractedVector0, BIPOLAR);
            std::vector<int> input_kernel_SCMultiplication = bitstreamOperation(SCtensor.getVectorAt(0,1), SCtensor.getVectorAt(0,0), XNOR);

        // Print the extracted vector
        std::cout << "Extracted vector at position (0, 2): ";
        for (int value : extractedVector0) {
            std::cout << value << " ";
        }

        std::cout << "Extracted vector at position (1, 1): ";
        for (int value : extractedVector1) {
            std::cout << value << " ";
        }

        std::cout << "Extracted vector at position (2, 0): ";
        for (int value : extractedVector2) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    } catch (const std::out_of_range& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}*/
