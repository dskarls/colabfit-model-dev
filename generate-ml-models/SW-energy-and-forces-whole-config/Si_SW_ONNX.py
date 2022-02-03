import sys

import numpy as np
import onnx
import onnxruntime

from load_training_set import load_kliff_training_set

# =============================================================================
# =============================================================================

if __name__ == "__main__":

    training_set_dir = sys.argv[1]

    load_onnx = onnx.load("SW_en.onnx")
    onnx.checker.check_model(load_onnx)

    ort_session = onnxruntime.InferenceSession("SW_en.onnx")

    # Load inputs for all configurations from training set
    desc, energies, forces = load_kliff_training_set(training_set_dir, 3.77118)

    # Pull out inputs for the first configuration
    particle_contributing, coords, num_neighbors, neighbor_list, *_ = desc[0][:]

    model_inputs = {
        "particle_contributing": particle_contributing,
        "coords": coords,
        "num_neighbors": num_neighbors,
        "neighbor_list": neighbor_list,
    }
    energy = ort_session.run(None, model_inputs)[0]

    print(f"Energy: {energy}")
