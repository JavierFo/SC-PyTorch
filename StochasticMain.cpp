#include "StochasticTensor.h"
#include <iostream>
#include <vector>
#include <iomanip>

// Function to pretty print the resulting 2D vector
void prettyPrint2D(const std::vector<std::vector<double>>& result) {
    std::cout << "{\n";
    for (const auto& row : result) {
        std::cout << "  {";
        for (size_t i = 0; i < row.size(); ++i) {
            std::cout << std::fixed << std::setprecision(2) << row[i];
            if (i < row.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}\n";
    }
    std::cout << "}\n";
}

// int main() {
//     //int x = 3, y = 3, z = 5; // Dimensions of the tensor
//     //StochasticTensor st(x, y, z);

//         // Example kernel matrix
//     std::vector<std::vector<int>> kernel = {
//         {-55, 113, 9},
//         {213, 0, 44},
//         {-185, 250, -255}
//     };
//     int N = 10000; // size of lfsr based random numbers / bitstream length
//     std::cout << "Random Arrays:\n";
//     //FOR UNIPOLAR REPRESENTATION:
//     //StochasticTensor SCtensor(kernel, N, MT19937);
//     //FOR BIPOLAR REPRESENTATION:
//     StochasticTensor SCtensor(kernel, N, MT19937, -255, 255);
//     //std::cout << "Stochastic Tensor:\n";
//     //SCtensor.printStochasticTensor();
//     //std::cout << "Original Tensor from Stochastic:\n";
//     prettyPrint2D(SCtensor.toRealTensor(255, BIPOLAR));
//     //StochasticTensor::SizeTuple sizes = SCtensor.getSize();
//     // try {
//     //     // Extract the vector at position (1,1)
//     //     std::vector<int> extractedVector0 = SCtensor.getVectorAt(0, 1);
//     //     std::vector<int> extractedVector1 = SCtensor.getVectorAt(1, 0);
//     //     std::vector<int> extractedVector2 = SCtensor.getVectorAt(2, 1);

//     //        double o = calculatePx(extractedVector0, BIPOLAR);
//     //         std::vector<int> input_kernel_SCMultiplication = bitstreamOperation(SCtensor.getVectorAt(0,1), SCtensor.getVectorAt(0,0), XNOR);

//     //     // Print the extracted vector
//     //     std::cout << "Extracted vector at position (0, 2): ";
//     //     for (int value : extractedVector0) {
//     //         std::cout << value << " ";
//     //     }

//     //     std::cout << "Extracted vector at position (1, 1): ";
//     //     for (int value : extractedVector1) {
//     //         std::cout << value << " ";
//     //     }

//     //     std::cout << "Extracted vector at position (2, 0): ";
//     //     for (int value : extractedVector2) {
//     //         std::cout << value << " ";
//     //     }
//     //     std::cout << std::endl;
//     // } catch (const std::out_of_range& e) {
//     //     std::cerr << e.what() << std::endl;
//     // }

//     return 0;
// }
