#include "ml_model.hpp"

#include <torch/script.h>

MLModel *MLModel::create(const char *model_file_path, MLModelType ml_model_type)
{
    if (ml_model_type == ML_MODEL_PYTORCH)
    {
        return new PytorchModel(model_file_path);
    }
}

void PytorchModel::SetInputNode(int model_input_index, int *input, int size,
                                bool requires_grad)
{
    // TODO: Move this if-else block for type checking to its own private
    // method? This would require making a template for it though and making
    // explicit instantiations for all possible types
    // TODO: Is it possible/appropriate to use a type cast for this?

    // Get the size used by 'int' on this platform and set torch tensor type
    // appropriately
    const std::size_t platform_size_int = sizeof(int);

    torch::TensorOptions tensor_options = torch::TensorOptions();

    if (platform_size_int == 1)
    {
        tensor_options.dtype(torch::kInt8);
    }
    else if (platform_size_int == 2)
    {
        tensor_options.dtype(torch::kInt16);
    }
    else if (platform_size_int == 4)
    {
        tensor_options.dtype(torch::kInt32);
    }
    else if (platform_size_int == 8)
    {
        tensor_options.dtype(torch::kInt64);
    }

    torch::Tensor input_tensor =
        torch::from_blob(input, {size}, tensor_options);

    if (requires_grad)
    {
        input_tensor.requires_grad_();
    }

    model_inputs_[model_input_index] = input_tensor;
}

void PytorchModel::SetInputNode(int model_input_index, double *input, int size,
                                bool requires_grad)
{
    // TODO: Move this if-else block for type checking to its own private
    // method? This would require making a template for it though and making
    // explicit instantiations for all possible types
    // TODO: Is it possible/appropriate to use a type cast for this?
    //
    // NOTE: Support for bool input types has not been added below because of
    // C++'s special handling of vectors of bools that causes various issues,
    // e.g. instantiating this method template with bool will fail.
    auto input_tensor =
        torch::from_blob(input, {size}, torch::dtype(torch::kFloat64));

    if (requires_grad)
    {
        input_tensor.requires_grad_();
    }

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

    // Reserve size for the four fixed model inputs (particle_contributing,
    // coordinates, number_of_neighbors, neighbor_list)
    model_inputs_.resize(4);

    // Set model to evaluation mode to set any dropout or batch normalization
    // layers to evaluation mode
    module_.eval();
}

PytorchModel::~PytorchModel() {}