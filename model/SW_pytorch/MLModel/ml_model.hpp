#ifndef MLMODEL_HPP
#define MLMODEL_HPP

#include <stdlib.h>
#include <vector>

#include <torch/script.h>

enum MLModelType
{
    ML_MODEL_PYTORCH,
};

/* Abstract base class for an ML model -- 'product' of the factory pattern */
class MLModel
{
public:
    static MLModel *create(const char * /*model_file_path*/,
                           MLModelType /*ml_model_type*/,
                           const char *const /*device_name*/);

    // TODO: Should we use named inputs instead?  I believe they're required
    // by ONNX, but not sure exactly how they work vis-a-vis exporting to a
    // torchscript file.

    // Function templates can't be used for pure virtual functions, and since
    // SetInputNode and Run each have their own (different) support argument
    // types, we can't use a class template.  So, we explicitly define each
    // supported overloading.
    virtual void SetInputNode(int /*model_input_index*/, int * /*input*/,
                              int /*size*/, bool requires_grad = false) = 0;
    virtual void SetInputNode(int /*model_input_index*/, double * /*input*/,
                              int /*size*/, bool requires_grad = false) = 0;

    virtual void Run(double * /*energy*/, double * /*forces*/) = 0;

    virtual ~MLModel(){};
};

// Concrete MLModel corresponding to pytorch
class PytorchModel : public MLModel
{
private:
    torch::jit::script::Module module_;
    std::vector<torch::jit::IValue> model_inputs_;
    torch::Device *device_;

    torch::Dtype get_torch_data_type(int *);
    torch::Dtype get_torch_data_type(double *);

    void SetExecutionDevice(const char * /*device_name*/);

public:
    const char *model_file_path_;

    PytorchModel(const char * /*model_file_path*/,
                 const char *const /*device_name*/);

    void SetInputNode(int /*model_input_index*/, int * /*input*/, int /*size*/,
                      bool requires_grad = false);
    void SetInputNode(int /*model_input_index*/, double * /*input*/,
                      int /*size*/, bool requires_grad = false);

    void Run(double * /*energy*/, double * /*forces*/);

    ~PytorchModel();
};

#endif /* MLMODEL_HPP */