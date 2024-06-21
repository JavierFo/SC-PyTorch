
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>

#include "StochasticTensor.h"

// Function to calculate the real number from a stochastic bitstream
double convertFromStochasticBitstream_(const std::vector<int>& bitstream) {
    // Calculate the probability of 1s
    double countOnes = std::count(bitstream.begin(), bitstream.end(), 1);
    double probability = static_cast<double>(countOnes) / bitstream.size();
    std::cout << "ONES: " << countOnes << std::endl;
    // Convert probability to bipolar value
    double bipolarValue = 2 * probability - 1;
    //std::cout << "bipolar val: " << bipolarValue << std::endl;

    // Denormalize to the original range
    //double realNumber = (bipolarValue + 1) / 2 * (maxRange - minRange) + minRange;

    return bipolarValue;
}

// Function to multiply two stochastic bitstreams element-wise
std::vector<int> multiplyStochasticBitstreams(const std::vector<int>& bitstream1, const std::vector<int>& bitstream2) {
    std::vector<int> result(bitstream1.size());
    for (size_t i = 0; i < bitstream1.size(); ++i) {
        //result[i] = bitstream1[i] & bitstream2[i];  // Logical AND to multiply
        result[i] = ~(bitstream1[i] ^ bitstream2[i]) & 1;
    }
    return result;
}

// int main() {
//     // Define the input real numbers
//     double input1 = .5;
//     double input2 = .9;
//     int bitstreamLength = 100;  // Length of the stochastic bitstream
//     double minRange = 0;
//     double maxRange = 255;
//     std::vector<int> bitstream1, bitstream2, resultBitstream;

//     // Convert to stochastic bitstreams
//     stochasticNumberGenerator(bitstreamLength, LFSR, input1, BIPOLAR, bitstream1);
//     stochasticNumberGenerator(bitstreamLength, LFSR, input2, BIPOLAR, bitstream2);

//     double scbitstream1 = convertFromStochasticBitstream_(bitstream1);
//         std::cout << "SC bitstream1: " << scbitstream1 << std::endl;

//     double scbitstream1_ = calculatePx(bitstream1 , BIPOLAR);
//         std::cout << "SC bitstream1_: " << scbitstream1_ << std::endl;

//     double scbitstream2 = convertFromStochasticBitstream_(bitstream2);
//         std::cout << "SC bitstream2: " << scbitstream2 << std::endl;

//     double scbitstream2_ = calculatePx(bitstream2 , BIPOLAR);
//         std::cout << "SC bitstream2_: " << scbitstream2_ << std::endl;

//     //std::cout << "SC bitstream2: " << scbitstream2 << std::endl;

//     // Multiply the two stochastic bitstreams
//     //resultBitstream = multiplyStochasticBitstreams(bitstream1, bitstream2);
//     // bitstream1 = {1,1,1,1,1,1,1,1};
//     // bitstream2 = {0,0,0,1,1,0,0,1};

//     resultBitstream = bitstreamOperation(bitstream1, bitstream2, XNOR);

//     //std::cout << "Multiplication: " << std::endl;
//     // Convert the result bitstream back to real form
//     double resultReal = convertFromStochasticBitstream_(resultBitstream);

//     std::cout << "Converted result real value: " << resultReal << std::endl;

//     double resultAddition = calculatePx(bitstream1, BIPOLAR, bitstream2)*2;
//     std::cout << "Addition result value: " << resultAddition << std::endl;
//     //std::cout << "Converted result real value: " << resultReal*255 << std::endl;

//     // Print the bitstreams and the result
//     // std::cout << "Bitstream for " << input1 << ": ";
//     // for (int bit : bitstream1) {
//     //     std::cout << bit;
//     // }
//     // std::cout << std::endl;

//     // std::cout << "Bitstream for " << input2 << ": ";
//     // for (int bit : bitstream2) {
//     //     std::cout << bit;
//     // }
//     // std::cout << std::endl;

//     std::cout << "Resulting bitstream: ";
//     for (int bit : resultBitstream) {
//         std::cout << bit;
//     }
//     std::cout << std::endl;

//     return 0;
// }