#include <iostream>
#include <bitset>

// Function to generate the next state of the LFSR
uint16_t lfsr_next_state(uint16_t state) {
    // XOR the bits according to the feedback polynomial
    uint16_t feedback = ((state >> 3) ^ (state >> 12) ^ (state >> 14) ^ (state >> 15)) & 1;
    // Shift right by one bit and set the leftmost bit as the feedback
    state = (state >> 1) | (feedback << 15);
    return state;
}

/*int main() {
    // Initial state of the LFSR (non-zero for maximum sequence length)
    uint16_t lfsr_state = 0b0101010101010101; // Example initial state
    // Convert bitset to integer
    
    // Generate and print the next 16 states of the LFSR
    for (int i = 0; i < 16; ++i) {
        std::cout << "State " << i + 1 << ": " << std::bitset<16>(lfsr_state) << std::endl;
        std::cout << "Integer value: " << lfsr_state << std::endl;
        lfsr_state = lfsr_next_state(lfsr_state);
        
    }

    return 0;
}*/
