#include <vector>
#include <iostream>
//STOCHASTIC TENSOR GENERATOR TEST/////////////////////////////////////////////////////////////////////
// Provided function
void stochasticNumberGenerator1(int *randomNumbers, int N, int externalNumber, std::vector<int> &output) {
    // Resize the output vector to ensure it has enough space
    output.resize(sizeof(randomNumbers));  
    for (int i = 0; i < sizeof(randomNumbers); ++i) {
        if (randomNumbers[i] < externalNumber) {
            output[i] = 1;
        } else {
            output[i] = 0;
        }
    }
}

// The new function that processes the input and generates the output
std::vector<std::vector<std::vector<int>>> processVector(
    const std::vector<std::vector<int>>& A, 
    int *randomNumbers, 
    int N
) {
    std::vector<std::vector<std::vector<int>>> B(A.size());

    for (size_t i = 0; i < A.size(); ++i) {
        B[i].resize(A[i].size());
        for (size_t j = 0; j < A[i].size(); ++j) {
            stochasticNumberGenerator1(randomNumbers, N, A[i][j], B[i][j]);
        }
    }

    return B;
}

/*int main() {
    // Example usage
    std::vector<std::vector<int>> A = {
        {5, 10, 15, 8, 3, 55, 22},
        {20, 25, 255, 260, 55, 6, 87},
        {4000, 2000, 6000, 2000, 5500, 1000, 300}
    };

    int randomNumbers[] = {1, 255, 10, 1500, 20, 25}; // Example random numbers
    int N = 6; // Length of the randomNumbers array

    std::vector<std::vector<std::vector<int>>> B = processVector(A, randomNumbers, N);

    // Print the result
    for (const auto& row : B) {
        for (const auto& vec : row) {
            for (int num : vec) {
                std::cout << num << " ";
            }
            std::cout << "\t";
        }
        std::cout << "\n";
    }

    return 0;
}*/
