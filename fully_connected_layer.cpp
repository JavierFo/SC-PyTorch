#include <iostream>
#include <vector>
#include <cmath>

class FullyConnectedLayer {
private:
    int numInputs;
    int numNeurons;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

public:
    FullyConnectedLayer(int inputSize, int numNeurons) : numInputs(inputSize), numNeurons(numNeurons) {
        // Initialize weights randomly
        for (int i = 0; i < numNeurons; ++i) {
            std::vector<double> neuronWeights;
            for (int j = 0; j < numInputs; ++j) {
                // Initialize weights randomly with values between -0.5 and 0.5
                neuronWeights.push_back((double)rand() / RAND_MAX - 0.5);
            }
            weights.push_back(neuronWeights);
        }

        // Initialize biases to zero
        biases = std::vector<double>(numNeurons, 0.0);
    }

    std::vector<double> forward(const std::vector<double>& inputs) {
        if (inputs.size() != numInputs) {
            std::cerr << "Error: Input size does not match layer size." << std::endl;
            return {};
        }

        std::vector<double> output(numNeurons, 0.0);

        // Compute output of each neuron
        for (int i = 0; i < numNeurons; ++i) {
            double neuronOutput = 0.0;
            for (int j = 0; j < numInputs; ++j) {
                neuronOutput += inputs[j] * weights[i][j];
            }
            neuronOutput += biases[i];
            output[i] = sigmoid(neuronOutput);
        }

        return output;
    }

    double computeLoss(const std::vector<double>& outputs, const std::vector<double>& targets) {
        if (outputs.size() != targets.size()) {
            std::cerr << "Error: Output size does not match target size." << std::endl;
            return -1;
        }

        double loss = 0.0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            double diff = outputs[i] - targets[i];
            loss += diff * diff;
        }
        return loss / outputs.size(); // Mean squared error
    }

private:
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }
};

/*int main() {
    // Example usage
    FullyConnectedLayer fcLayer(2, 3); // Input size: 2, Number of neurons: 3

    std::vector<double> inputs = {0.5, 0.7};
    std::vector<double> targets = {1.0, 0.0, 1.0}; // Example target output

    std::vector<double> output = fcLayer.forward(inputs);

    std::cout << "Output: ";
    for (auto val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    double loss = fcLayer.computeLoss(output, targets);
    std::cout << "Loss: " << loss << std::endl;

    return 0;
}*/
