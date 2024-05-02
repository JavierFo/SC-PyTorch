#include <iostream>
#include <cstdlib> // For rand() function
#include <ctime>   // For time() function

void compareAndSave(int *randomNumbers, int N, int B, int8_t *output) {
    for (int i = 0; i < N; ++i) {
        if (randomNumbers[i] < B) {
            output[i] = 1;
        } else {
            output[i] = 0;
        }
    }
}

int main() {
    const int N = 10; // Number of random numbers
    int randomNumbers[N];
    int8_t output[N];

    // Seed the random number generator
    srand(time(nullptr));

    // Generate random numbers
    for (int i = 0; i < N; ++i) {
        randomNumbers[i] = rand() % 100; // Generating random numbers between 0 and 99
        std::cout << randomNumbers[i] << " ";
    }
    std::cout << std::endl;

    int B = 44;

    // Compare random numbers to B and save the result in the output array
    compareAndSave(randomNumbers, N, B, output);

    // Output the comparison result
    std::cout << "Comparison result: ";
    for (int i = 0; i < N; ++i) {
        std::cout << static_cast<int>(output[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}
