# TODO: Do we need to impose a minimum cmake version here?

add_library(MLModel STATIC ml_model.cpp)

# FIXME: This is a temporary hack so I don't have to formally install torch
# into system directories.  Delete this later.
list(APPEND CMAKE_PREFIX_PATH "/root/libtorch/libtorch/")

find_package(Torch REQUIRED)
target_link_libraries(MLModel PRIVATE ${TORCH_LIBRARIES})

# TODO: Set CXX_STANDARD target property here?

# TODO: Copy DLLs if on windows?  See
# https://pytorch.org/cppdocs/installing.html#minimal-example
