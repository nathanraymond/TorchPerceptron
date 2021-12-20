#pragma once
#include <torch/torch.h>
#include <iostream>

class Perceptron {
public:
	Perceptron(int number_of_inputs);
	torch::Tensor bias = torch::ones(1);
	torch::Tensor weights;
	void set_weights(std::vector<double> w_init);
	double step_function(torch::Tensor x);
	double run(std::vector<double> x);
	double output(torch::Tensor x);
};
