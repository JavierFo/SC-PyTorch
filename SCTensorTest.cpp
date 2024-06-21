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

void prettyPrint3DVector(const std::vector<std::vector<std::vector<double>>>& vec, int precision = 5) {
  // Loop through each depth slice
  for (size_t depth = 0; depth < vec.size(); ++depth) {
    std::cout << "Depth Slice " << depth << ":" << std::endl;

    // Loop through each row (2D vector) in the slice
    for (size_t row = 0; row < vec[depth].size(); ++row) {
      // Print each element in the row
      for (size_t col = 0; col < vec[depth][row].size(); ++col) {
        std::cout << std::fixed << std::setprecision(precision)
                  << vec[depth][row][col] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

// int main() {

//     using namespace std;
//     using Tensor3D = vector<vector<vector<double>>>;

//         // Example kernel matrix
//     std::vector<std::vector<int>> kernel = {
//         {-55, 113, 9},
//         {213, 0, 44},
//         {-185, 250, -255}
//     };
//     int N = 10000; // size of lfsr based random numbers / bitstream length

//     Tensor3D input = {
//         {
//             {.1, -.2, -.3, -.4, .5}
//         },
//         {
//             {-.26, -.27, .28, .29, .30},
//             {-.31, -.32, .33, .34, .35}
//         },
//         {
//             {-.51, -.52, .53, .54, .55},
//             {-.56, .57, -.58, .59, -.60},
//             {.61, .62, .63, -.64, -.65}
//         }
//     };

//     //FOR UNIPOLAR REPRESENTATION:
//     //StochasticTensor SCtensor(kernel, N, MT19937);
//     //FOR BIPOLAR REPRESENTATION:
//     StochasticTensor SCtensor(input, N, MT19937, BIPOLAR);
//     //std::cout << "Stochastic Tensor:\n";
//     //SCtensor.printStochasticTensor();
//     //std::cout << "Original Tensor from Stochastic:\n";
//     prettyPrint3DVector(SCtensor.toReal3DTensor(1, BIPOLAR));
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

