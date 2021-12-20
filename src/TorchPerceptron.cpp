#include <torch/torch.h>
#include <iostream>
#include "TorchPerceptron.h"

Perceptron::Perceptron(int number_of_inputs) {
	this->bias = bias; // Save bias value
}

void Perceptron::set_weights(std::vector<double> w_init) {
	weights = torch::tensor(w_init, { torch::kFloat64 });
}

double Perceptron::run(std::vector<double> x) {
	return output(torch::tensor(x, { torch::kFloat64 }));
}

double Perceptron::output(torch::Tensor x) {
	auto inputs = torch::cat({ x, bias }, 0); // Add bias term to the end of the inputs vector
	auto weighted_sum = torch::dot(inputs, weights); // Dot product of Inputs⋅Weights vectors
	return step_function(weighted_sum);
}


double Perceptron::step_function(torch::Tensor weighted_sum) {
	// weighted_sum is a 1D tensor
	if (weighted_sum.item<int>() >= 0) {
		return 1;
	}
	else {
		return 0;
	}
}



int main() {
	try {
		Perceptron* p = new Perceptron(2);



		/*    OR GATE*/

		p->set_weights({ 15,15,-10 });

		std::cout << "Output: " << std::endl;
		std::cout << p->run({ 0,0 }) << std::endl;
		std::cout << p->run({ 0,1 }) << std::endl;
		std::cout << p->run({ 1,0 }) << std::endl;
		std::cout << p->run({ 1,1 }) << std::endl;

		/*   AND GATE*/

	/*	p->set_weights({ 10,10,-15 });

		std::cout << "Output: " << std::endl;
		std::cout << p->run({ 0,0 }) << std::endl;
		std::cout << p->run({ 0,1 }) << std::endl;
		std::cout << p->run({ 1,0 }) << std::endl;
		std::cout << p->run({ 1,1 }) << std::endl;*/

		std::cin.get();
	}

	catch (const c10::Error& e)
	{
		std::cout << e.msg() << std::endl;
	}

}