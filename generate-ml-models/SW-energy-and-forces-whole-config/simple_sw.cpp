#include<iostream>
#include<vector>
#include<fstream>
#include<torch/script.h>
#include<torch/csrc/autograd/autograd.h>

using namespace std;

int main(int argc, char const *argv[])
{
    string data_folder, model_name;
    vector<int> neighbor_list, num_neighbors, particle_contributing;
    vector<double> coords;
    fstream iofile;
    double num1;
    int num2;
    // ----------------------------------------------------
    torch::jit::script::Module model;
    torch::Tensor nl_tensor, nn_tensor, pc_tensor;
    torch::Tensor coords_tensor;
    // ----------------------------------------------------


    // Check for arguments
    if (argc <3
     ){
        cout<<"Provide folder containing files for coords, neighbor_list, num_neighbors, and particle_contributing\n and the pt model"<<endl;
        return -1;
    } else {
        data_folder = argv[1];
        model_name = argv[2];
    }

    cout << "Reading data from " << data_folder << endl;
    cout << "Loading Model" << model_name << endl;

    // Load data
    iofile.open(data_folder+"/coords.txt");
    iofile >> num1;
    while (!iofile.eof()) {
        coords.push_back(num1);
        iofile >> num1;
    }
    iofile.close();

    iofile.open(data_folder+"/neighbor_list.txt");
    iofile >> num2;
    while (!iofile.eof()) {
        neighbor_list.push_back(num2);
        iofile >> num2;
    }
    iofile.close();

    iofile.open(data_folder+"/num_neighbors.txt");
    iofile >> num2;
    while (!iofile.eof()) {
        num_neighbors.push_back(num2);
        iofile >> num2;
    }
    iofile.close();


    iofile.open(data_folder+"/particle_contributing.txt");
    iofile >> num2;
    while (!iofile.eof()) {
        particle_contributing.push_back(num2);
        iofile >> num2;
    }
    iofile.close();



    try
    {
        model = torch::jit::load(argv[2]);
    }
    catch(const c10::Error& e)
    {
        std::cerr << "Error in loading model" << '\n';
        return -1;
    }
    cout << "Model loaded" <<endl;


    // convert loaded vectors to tensors
    auto opts = torch::TensorOptions().dtype(torch::kInt32);

    nl_tensor = torch::from_blob(neighbor_list.data(), neighbor_list.size(), opts);
    nn_tensor = torch::from_blob(num_neighbors.data(), num_neighbors.size(), opts);
    pc_tensor = torch::from_blob(particle_contributing.data(), particle_contributing.size(), opts);
    opts = torch::TensorOptions().dtype(torch::kFloat64);
    coords_tensor = torch::from_blob(coords.data(), coords.size(), opts);
    coords_tensor.requires_grad_(true);
    cout<<"Gradients Enabled: " <<coords_tensor.requires_grad()<< endl;

    vector<torch::jit::IValue> model_inputs;
    model_inputs.push_back(pc_tensor);
    model_inputs.push_back(coords_tensor);
    model_inputs.push_back(nn_tensor);
    model_inputs.push_back(nl_tensor);


    //---------------------------------------------------------------
    //IValue tensor workaroud

    // Forward pass and gradients
    auto output = model.forward(model_inputs);
    auto forces = output.toTuple()->elements()[1]; // get forces out
    // based on https://github.com/pytorch/pytorch/issues/22440
    //----------------------------------------------------------------

    cout << forces <<"\n";
    return 0;
}
