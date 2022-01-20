#
# CDDL HEADER START
#
# The contents of this file are subject to the terms of the Common Development
# and Distribution License Version 1.0 (the "License").
#
# You can obtain a copy of the license at
# http://www.opensource.org/licenses/CDDL-1.0.  See the License for the
# specific language governing permissions and limitations under the License.
#
# When distributing Covered Code, include this CDDL HEADER in each file and
# include the License file in a prominent location with the name LICENSE.CDDL.
# If applicable, add the following below this CDDL HEADER, with the fields
# enclosed by brackets "[]" replaced with your own identifying information:
#
# Portions Copyright (c) [yyyy] [name of copyright owner]. All rights reserved.
#
# CDDL HEADER END
#

#
# Copyright (c) 2021 Regents of the University of Minnesota.
# All rights reserved.
#
# Contributors:
#    Daniel S. Karls
#

cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})
find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

kim_api_items_setup_before_project(ITEM_TYPE "portableModel")
project(SW_pytorch LANGUAGES CXX)
kim_api_items_setup_after_project(ITEM_TYPE "portableModel")

# This model driver requires C++11
set(CMAKE_CXX_STANDARD 14)

add_kim_api_model_library(
  NAME                    ${PROJECT_NAME}
  CREATE_ROUTINE_NAME     "model_create"
  CREATE_ROUTINE_LANGUAGE "cpp"
  )

add_subdirectory(MLModel)

target_sources(${PROJECT_NAME} PRIVATE
  SW_pytorch.cpp
)

find_package(Torch REQUIRED)
target_link_libraries("${PROJECT_NAME}" PRIVATE MLModel PRIVATE ${TORCH_LIBRARIES})

target_include_directories("${PROJECT_NAME}" PRIVATE
  "${PROJECT_BINARY_DIR}"
  "${PROJECT_SOURCE_DIR}/MLModel"
  )
