#include <iostream>
// #include <cstdlib> // For rand() function
// #include <ctime>   // For time() function
// #include "StochasticTensor.h"
#include <vector>
#include <stdexcept>

/*int main() {
    const int N = 10; // Number of random numbers
    std::vector<int> randomNumbers(N);
    std::vector<int> output;

    uint8_t lfsr_state = 0b01010101; // Example initial state

    for (int i = 0; i < N; ++i) {
        randomNumbers[i] = std::bitset<8>(lfsr_state).to_ulong(); 
        lfsr_state = LFSR_StatesGenerator(lfsr_state);
        std::cout << randomNumbers[i] << " ";
    }

    int B = 44;

    // Compare random numbers to B and save the result in the output array
    stochasticNumberGenerator(randomNumbers, B, output);

    // Output the comparison result
    std::cout << "Comparison result: ";
    for (int i = 0; i < N; ++i) {
       std::cout << static_cast<int>(output[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}*/
