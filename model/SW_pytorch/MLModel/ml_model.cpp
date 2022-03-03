#include <map>

#include "ml_model.hpp"

#include <torch/script.h>

MLModel *MLModel::create(const char *model_file_path, MLModelType ml_model_type,
                         const char *device_name)
{
    if (ml_model_type == ML_MODEL_PYTORCH)
    {
        return new PytorchModel(model_file_path, device_name);
    }
    // FIXME: raise an exception here if ``ml_model_type`` doesn't match any
    // known enumerations
}

void PytorchModel::SetExecutionDevice(const char *device_name)
{ // Use the requested device name char array to create a torch Device object

    // Default to 'cpu'
    if (device_name == nullptr)
    {
        device_name = "cpu";
    }

    torch::Device torch_device(device_name);
    device_ = &torch_device;
};

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

// TODO: Find a way to template SetInputNode...there are multiple definitions
// below that are exactly the same.  Since even derived implementations are
// virtual functions are also virtual, we can't use regular C++ templates.  Is
// it worth using the preprocessor for this?
void PytorchModel::SetInputNode(int model_input_index, int *input, int size,
                                bool requires_grad)
{
    // Map C++ data type used for the input here into the appropriate
    // fixed-width torch data type
    torch::Dtype torch_dtype = get_torch_data_type(input);

    // FIXME: Determine device to create tensor on
    torch::TensorOptions tensor_options =
        torch::TensorOptions().dtype(torch_dtype).requires_grad(requires_grad);

    // Finally, create the input tensor and store it on the relevant MLModel
    // attr
    torch::Tensor input_tensor =
        torch::from_blob(input, {size}, tensor_options).to(*device_);

    model_inputs_[model_input_index] = input_tensor;
}

void PytorchModel::SetInputNode(int model_input_index, double *input, int size,
                                bool requires_grad)
{
    // Map C++ data type used for the input here into the appropriate
    // fixed-width torch data type
    torch::Dtype torch_dtype = get_torch_data_type(input);

    // FIXME: Determine device to create tensor on
    torch::TensorOptions tensor_options =
        torch::TensorOptions().dtype(torch_dtype).requires_grad(requires_grad);

    // Finally, create the input tensor and store it on the relevant MLModel
    // attr
    torch::Tensor input_tensor =
        torch::from_blob(input, {size}, tensor_options).to(*device_);

    model_inputs_[model_input_index] = input_tensor;
}

void PytorchModel::Run(double *energy, double *forces)
{
    // FIXME: Make this work for arbitrary number/type of outputs?  This may
    // lead us to make Run() take no parameters, and instead define separate
    // methods for accessing each of the outputs of the ML model.

    // Run ML model's `forward` method and retrieve outputs as tuple
    // IMPORTANT: We require that the pytorch model's `forward`
    // method return a tuple where the energy is the first entry and
    // the forces are the second
    const auto output_tensor_list =
        module_.forward(model_inputs_).toTuple()->elements();

    // After moving the first output tensor back to the CPU (if necessary),
    // extract its value as the partial energy
    *energy =
        *output_tensor_list[0].toTensor().to(torch::kCPU).data_ptr<double>();

    // After moving the second output tensor back to the CPU (if necessary),
    // extract its contents as the partial forces
    auto torch_forces = output_tensor_list[1].toTensor().to(torch::kCPU);

    // TODO: Move the accessor data extraction to a separate private method
    auto force_accessor = torch_forces.accessor<double, 1>();
    for (int atom_count = 0; atom_count < force_accessor.size(0); ++atom_count)
    {
        forces[atom_count] = force_accessor[atom_count];
    }
}

PytorchModel::PytorchModel(const char *model_file_path, const char *device_name)
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

    SetExecutionDevice(device_name);

    // Copy model to execution device
    module_.to(*device_);

    // Reserve size for the four fixed model inputs (particle_contributing,
    // coordinates, number_of_neighbors, neighbor_list)
    model_inputs_.resize(4);

    // Set model to evaluation mode to set any dropout or batch normalization
    // layers to evaluation mode
    module_.eval();
}

PytorchModel::~PytorchModel() {}