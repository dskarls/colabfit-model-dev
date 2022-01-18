#include "ml_model.hpp"

//#include <torch/script.h>
#include "torch/script.h"

MLModel *MLModel::create(const char *model_file_path, MLModelType ml_model_type)
{
    if (ml_model_type == ML_MODEL_PYTORCH)
    {
        return new PytorchModel(model_file_path);
    }
}

template <typename input_tensor_element_type>
void PytorchModel::PushInputNode(input_tensor_element_type *input)
{
    // TODO: Move this if-else block for type checking to its own private
    // method? This would require making a template for it though and making
    // explicit instantiations for all possible types
    // TODO: Is it possible/appropriate to use a type cast for this?
    //
    // NOTE: Support for bool input types has not been added below because of
    // C++'s special handling of vectors of bools that causes various issues,
    // e.g. instantiating this method template with bool will fail.
    if (std::is_same<input_tensor_element_type, uint8_t>::value)
    {
        auto input_tensor = torch::tensor(*input, torch::dtype(torch::kUInt8));
    }
    else if (std::is_same<input_tensor_element_type, int8_t>::value)
    {
        auto input_tensor = torch::tensor(*input, torch::dtype(torch::kInt8));
    }
    else if (std::is_same<input_tensor_element_type, int16_t>::value)
    {
        auto input_tensor = torch::tensor(*input, torch::dtype(torch::kInt16));
    }
    else if (std::is_same<input_tensor_element_type, int32_t>::value)
    {
        auto input_tensor = torch::tensor(*input, torch::dtype(torch::kInt32));
    }
    else if (std::is_same<input_tensor_element_type, int64_t>::value)
    {
        auto input_tensor = torch::tensor(*input, torch::dtype(torch::kInt64));
    }
    else if (std::is_same<input_tensor_element_type, float>::value)
    {
        auto input_tensor =
            torch::tensor(*input, torch::dtype(torch::kFloat32));
    }
    else if (std::is_same<input_tensor_element_type, double>::value)
    {
        auto input_tensor =
            torch::tensor(*input, torch::dtype(torch::kFloat64));
    }

    inputs_.push_back(input_tensor);
}

template <typename output_arr_type>
void PytorchModel::Run(output_arr_type *output)
{
    // Run model's `forward` method and convert the IValue returned to a tensor
    torch::Tensor output_tensor = model_->forward(inputs_).toTensor();

    // FIXME: Make this work for more than a single scalar output -- each
    // output can presumably have a different type, too.  This may lead us to
    // make Run() take no parameters, and instead define separate methods for
    // accessing each of the outputs of the model.
    auto output_accessor = output_tensor.accessor<output_arr_type, 0>();
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
        return -1;
    }

    // Set model to evaluation mode to set any dropout or batch normalization
    // layers to evaluation mode
    module_.eval();
}

PytorchModel::~PytorchModel() {}

// Instantiate templates for setting input data to pytorch model
template void PytorchModel::PushInputNode<uint8_t>(uint8_t *);
template void PytorchModel::PushInputNode<int8_t>(int8_t *);
template void PytorchModel::PushInputNode<int16_t>(int16_t *);
template void PytorchModel::PushInputNode<int32_t>(int32_t *);
template void PytorchModel::PushInputNode<int64_t>(int64_t *);
template void PytorchModel::PushInputNode<float>(float *);
template void PytorchModel::PushInputNode<double>(double *);

// Instantiate templates for setting output data from pytorch model
template void PytorchModel::Run<float>(float *);
template void PytorchModel::Run<double>(double *);
template void PytorchModel::Run<uint8_t>(uint8_t *);
template void PytorchModel::Run<int8_t>(int8_t *);
template void PytorchModel::Run<int16_t>(int16_t *);
template void PytorchModel::Run<int32_t>(int32_t *);
template void PytorchModel::Run<int64_t>(int64_t *);