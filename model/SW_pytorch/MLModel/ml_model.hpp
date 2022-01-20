#ifndef MLMODEL_HPP
#define MLMODEL_HPP

#include <stdlib.h>
#include <iostream>

#include <torch/script.h>

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

    // Function templates can't be used for pure virtual functions, and since
    // PushInputNode and Run each have their own (different) support argument
    // types, we can't use a class template.  So, we explicitly define each
    // supporting overloading.
    virtual void PushInputNode(int32_t *) = 0;
    virtual void PushInputNode(double *) = 0;

    virtual void Run(double*) = 0;
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

    virtual void PushInputNode(int32_t *);
    virtual void PushInputNode(double*);

    virtual void Run(double*);

    ~PytorchModel();
};

#endif /* MLMODEL_HPP */