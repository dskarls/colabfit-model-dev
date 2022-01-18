#ifndef MLMODEL_HPP
#define MLMODEL_HPP

#include <stdlib.h>
#include <iostream>

enum MLModelType
{
    ML_MODEL_PYTORCH,
};

/* Abstract base class for an ML model -- 'product' of the factory pattern */
class MLModel
{
public:
    static MLModel *create(const char *, MLModelType);

    // TODO: Should we use named inputs instead?  I believe they're required
    // by ONNX, but not sure exactly how they work vis-a-vis exporting to a
    // torchscript file.
    template <typename input_tensor_element_type>
    virtual void PushInputNode(input_tensor_element_type *) = 0;

    template <typename output_arr_type>
    virtual void Run(output_arr_type **) = 0;
};

// Concrete MLModel corresponding to pytorch
class PytorchModel : public MLModel
{
private:
    torch::jit::script::Module module_;
    std::vector<torch::jit::IValue> inputs_;

public:
    const char *model_file_path_;

    PytorchModel(const char *);

    template <typename input_tensor_element_type>
    void PushInputNode(input_tensor_element_type *);

    template <typename output_arr_type> void Run(output_arr_type **);
}

#endif /* MLMODEL_HPP */