// Based on https://pytorch.org/tutorials/advanced/cpp_export.html

#include <torch/script.h> // One-stop header.

#include <iostream>

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "Successfully loaded pytorch model\n";

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    auto A = torch::tensor({1.4}, torch::dtype(torch::kFloat64));
    auto B = torch::tensor({2.3}, torch::dtype(torch::kFloat64));
    inputs.push_back(A);
    inputs.push_back(B);

    // Execute the model and turn its output into a tensor.
    torch::Tensor output = module.forward(inputs).toTensor();

    std::cout << output << '\n';
}
