#include <iostream>
#include <array>
#include <random>

class StochasticTensor {
private:
    static constexpr int N = 4; // Number of elements in each array
    static constexpr int M = 1;  // Number of dimensions
    using ArrayType = std::array<int8_t, N>; // Define array type

    // Nested array structure to represent 3D tensor
    ArrayType data[M][N];

public:
// Method to generate random array with 12 1s and 0s
    ArrayType generateRandomArray() {
        ArrayType randomArray;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 1);

        for (int i = 0; i < N; ++i) {
            randomArray[i] = dis(gen);
        }

        return randomArray;
    }

        // Method to populate the tensor with random arrays
    void populateWithRandomArrays() {
        for (int k = 0; k < M; ++k) {
            for (int i = 0; i < N; ++i) {
                data[k][i] = generateRandomArray();
        }
    }
    }


    // Method to perform "multiplication" operation
    void multiply(const StochasticTensor& other) {
        for (int k = 0; k < M; ++k) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    data[k][i][j] &= other.data[k][i][j]; // AND operation
                }
            }
        }
    }

    // Method to perform "addition" operation
    void add(const StochasticTensor& other) {
        constexpr std::array<int8_t, N> selectInput = {1,0,0,0}; // MUX select input

        for (int k = 0; k < M; ++k) {
            for (int i = 0; i < N; ++i) {
                // Use MUX select input to choose between elements of A and B
                data[k][i] = (selectInput[i] == 1) ? data[k][i] : other.data[k][i];
            }
        }
    }

   // Method to print the tensor
    void print() const {
        for (int k = 0; k < M; ++k) {
        std::cout << "Dimension " << k + 1 << ":" << std::endl;
        std::cout << "[";
        for (int i = 0; i < N; ++i) {
            std::cout << "[";
            for (int j = 0; j < N; ++j) {
                std::cout << static_cast<int>(data[k][i][j]);
                if (j < N - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]";
            if (i < N - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    }
};

int main() {
    StochasticTensor A, B;

 // Populate tensors with random arrays
    A.populateWithRandomArrays();
    B.populateWithRandomArrays();

    // Print the tensors
    std::cout << "Tensor A:" << std::endl;
    A.print();
    std::cout << std::endl;

    std::cout << "Tensor B:" << std::endl;
    B.print();
    std::cout << std::endl;
    // Perform multiplication
    A.multiply(B);

    // Perform addition
    //A.add(B);

    // Print the resulting tensor
    std::cout << "Tensor A*B:" << std::endl;
    A.print();

    return 0;
}
