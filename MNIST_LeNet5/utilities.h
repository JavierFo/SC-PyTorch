#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <bitset>
#include <tuple>
#include <algorithm>
#include <cstdlib>

std::vector<std::vector<std::vector<uint8_t>>> convertToUint8(const std::vector<std::vector<std::vector<int>>>& input);

void prettyPrint2D(const std::vector<std::vector<double>>& matrix);

std::vector<float> generateRandomVector(size_t size);

void printTensor(const std::vector<std::vector<std::vector<double>>>& tensor);

std::vector<std::vector<std::vector<std::vector<uint8_t>>>> create4DTensor(const std::vector<std::vector<std::vector<double>>>& U, const int R, const std::vector<int>& output);

std::vector<std::vector<double>> generateRandomMatrix(int NoR, int NoC);

std::vector<std::vector<double>> generateRandomInputs(int NoR, int NoC);

size_t getTotalNumberOfElements(const std::vector<std::vector<std::vector<double>>>& vec);

size_t getTotalNumberOfElements2D(const std::vector<std::vector<double>>& vec);

std::vector<std::vector<double>> conv2_(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& kernel);

std::vector<double> generateRandom1DInputs(size_t size);

std::vector<float> n_forward(const std::vector<float>& inputs, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias);

std::vector<int8_t> castVectorToInt8(const std::vector<int>& input);

#endif // 