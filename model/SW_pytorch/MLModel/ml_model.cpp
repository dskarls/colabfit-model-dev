#include <map>

#include "ml_model.hpp"

#include <torch/script.h>

MLModel *MLModel::create(const char *model_file_path, MLModelType ml_model_type)
{
    if (ml_model_type == ML_MODEL_PYTORCH)
    {
        return new PytorchModel(model_file_path);
    }
}

torch::Dtype PytorchModel::get_torch_data_type(int *)
{
    // Get the size used by 'int' on this platform and set torch tensor type
    // appropriately
    const std::size_t platform_size_int = sizeof(int);

    std::map<int, torch::Dtype> platform_size_int_to_torch_dtype;

    platform_size_int_to_torch_dtype[1] = torch::kInt8;
    platform_size_int_to_torch_dtype[2] = torch::kInt16;
    platform_size_int_to_torch_dtype[4] = torch::kInt32;
    platform_size_int_to_torch_dtype[8] = torch::kInt64;

    torch::Dtype torch_dtype =
        platform_size_int_to_torch_dtype[platform_size_int];

    return torch_dtype;
}

torch::Dtype PytorchModel::get_torch_data_type(double *)
{
    return torch::kFloat64;
}

void PytorchModel::SetInputNode(int model_input_index, int *input, int size,
                                bool requires_grad)
{
    // Map C++ data type used for the input here into the appropriate
    // fixed-width torch data type
    torch::Dtype torch_dtype = get_torch_data_type(input);

    // FIXME: Determine device to create tensor on
    torch::Device torch_device(torch::kCPU);
    torch::TensorOptions tensor_options = torch::TensorOptions()
                                              .dtype(torch_dtype)
                                              .requires_grad(requires_grad)
                                              .device(torch_device);

    // Finally, create the input tensor and store it on the relevant MLModel
    // attr
    torch::Tensor input_tensor =
        torch::from_blob(input, {size}, tensor_options);

    model_inputs_[model_input_index] = input_tensor;
}

void PytorchModel::SetInputNode(int model_input_index, double *input, int size,
                                bool requires_grad)
{
    // Map C++ data type used for the input here into the appropriate
    // fixed-width torch data type
    torch::Dtype torch_dtype = get_torch_data_type(input);

    // FIXME: Determine device to create tensor on
    torch::Device torch_device(torch::kCPU);
    torch::TensorOptions tensor_options = torch::TensorOptions()
                                              .dtype(torch_dtype)
                                              .requires_grad(requires_grad)
                                              .device(torch_device);

    // Finally, create the input tensor and store it on the relevant MLModel
    // attr
    torch::Tensor input_tensor =
        torch::from_blob(input, {size}, tensor_options);

    model_inputs_[model_input_index] = input_tensor;
}

void PytorchModel::Run(double *energy, double *forces)
{
    // Run ML model's `forward` method and retrieve outputs as tuple
    const auto output_tensor_list =
        module_.forward(model_inputs_).toTuple()->elements();

    // Copy value of energy from first tensor outputted by model
    *energy = *output_tensor_list[0].toTensor().data_ptr<double>();

    auto torch_forces = output_tensor_list[1].toTensor();
    auto force_accessor = torch_forces.accessor<double, 1>();
    for (int atom_count = 0; atom_count < force_accessor.size(0); ++atom_count)
    {
        forces[atom_count] = force_accessor[atom_count];
    }
    // torch::Tensor output_tensor = module_.forward(model_inputs_).toTuple();

    // FIXME: Make this work for more than a single scalar output -- each
    // output can presumably have a different type, too.  This may lead us to
    // make Run() take no parameters, and instead define separate methods for
    // accessing each of the outputs of the ML model.
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
                     "pytorch model file from path "
                  << model_file_path << std::endl;
    }

    // FIXME: Determine device to copy model to
    torch::Device torch_device(torch::kCPU);
    module_.to(torch_device);

    // Reserve size for the four fixed model inputs (particle_contributing,
    // coordinates, number_of_neighbors, neighbor_list)
    model_inputs_.resize(4);

    // Set model to evaluation mode to set any dropout or batch normalization
    // layers to evaluation mode
    module_.eval();
}

PytorchModel::~PytorchModel() {}