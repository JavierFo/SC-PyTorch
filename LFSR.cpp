#include <iostream>

class LFSR {
private:
    unsigned char state; // The state of the LFSR

public:
    LFSR(unsigned char seed) : state(seed) {}

    // Function to generate the next random number
    unsigned char getNext() {
        // XOR the bits at specific positions to produce the next bit
        unsigned char next_bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1;
        
        // Shift the state to the right by 1 bit and insert the next bit
        state = (state >> 1) | (next_bit << 7);
        
        return state;
    }

    // Function to print the binary representation of a number
    void printBinary(unsigned char num) {
        for (int i = 7; i >= 0; --i) {
            std::cout << ((num >> i) & 1);
        }
        std::cout << std::endl;
    }
};

int main() {
    // Create an instance of the LFSR with an initial seed value
    LFSR lfsr(0x92u);

    // Generate and print the next 10 random numbers in binary representation
    for (int i = 0; i < 10; ++i) {
        unsigned char random_number = lfsr.getNext();
        std::cout << "Random number " << i + 1 << ": ";
        lfsr.printBinary(random_number);
    }

    return 0;
}