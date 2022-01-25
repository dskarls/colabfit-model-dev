#include "ml_model.hpp"

#include <torch/script.h>

MLModel *MLModel::create(const char *model_file_path, MLModelType ml_model_type)
{
    if (ml_model_type == ML_MODEL_PYTORCH)
    {
        return new PytorchModel(model_file_path);
    }
}

void PytorchModel::PushInputNode(int *input, int size)
{
    // TODO: Move this if-else block for type checking to its own private
    // method? This would require making a template for it though and making
    // explicit instantiations for all possible types
    // TODO: Is it possible/appropriate to use a type cast for this?

    // FIXME: This will fail if the compiler implementation isn't using 4-byte
    // integers.  Should use sizeof(int) here with some logic here
    auto input_tensor = torch::from_blob(input, {size}, torch::dtype(torch::kInt32));

    inputs_.push_back(input_tensor);
}

void PytorchModel::PushInputNode(double *input, int size)
{
    // TODO: Move this if-else block for type checking to its own private
    // method? This would require making a template for it though and making
    // explicit instantiations for all possible types
    // TODO: Is it possible/appropriate to use a type cast for this?
    //
    // NOTE: Support for bool input types has not been added below because of
    // C++'s special handling of vectors of bools that causes various issues,
    // e.g. instantiating this method template with bool will fail.
    auto input_tensor = torch::from_blob(input, {size}, torch::dtype(torch::kFloat64));

    inputs_.push_back(input_tensor);
}

void PytorchModel::Run(double *output)
{
    // Run ML model's `forward` method and convert the IValue returned to a
    // tensor
    torch::Tensor output_tensor = module_.forward(inputs_).toTensor();

    // FIXME: Make this work for more than a single scalar output -- each
    // output can presumably have a different type, too.  This may lead us to
    // make Run() take no parameters, and instead define separate methods for
    // accessing each of the outputs of the ML model.
    auto output_accessor = output_tensor.accessor<double, 1>();
    *output = output_accessor[0];
}

PytorchModel::PytorchModel(const char *model_file_path)
{
    model_file_path_ = model_file_path;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module_ = torch::jit::load(model_file_path_);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "ERROR: An error occurred while attempting to load the "
                     "pytorch model file"
                  << std::endl;
    }

    // Set model to evaluation mode to set any dropout or batch normalization
    // layers to evaluation mode
    module_.eval();
}

PytorchModel::~PytorchModel() {}