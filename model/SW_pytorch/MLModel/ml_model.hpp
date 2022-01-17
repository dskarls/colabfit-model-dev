#ifndef MLMODEL_HPP
#define MLMODEL_HPP

#include <stdlib.h>

#include <torch/script.h>

/* Main wrapper class for an ML model */
class MLModel
{
private:
public:
    // Contains the path to the serialized ML model file to load
    const char *model_path_;

    // TODO: Set up logging capabilities and have a log file path as a public
    // attribute

    // Constructor
    MLModel();

    template <typename input_tensor_element_type>
    void
    SetModelInputNodeTensorData(size_t input_node_index,
                                input_tensor_element_type *input_tensor_values);

    // Destructor
    ~MLModel();

    void Run();
};

#endif /* MLMODEL_HPP */