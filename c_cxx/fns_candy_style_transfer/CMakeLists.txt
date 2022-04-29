# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_executable(fns_candy_style_transfer "fns_candy_style_transfer.c")
target_include_directories(fns_candy_style_transfer PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS})
target_link_libraries(fns_candy_style_transfer PRIVATE onnxruntime ${PNG_LIBRARIES})
if(PNG_LIBDIR)
  target_link_directories(fns_candy_style_transfer PRIVATE ${PNG_LIBDIR})
endif()

find_package(CUDA REQUIRED)

add_executable(fns_candy_style_transfer_cpp "fns_candy_style_transfer.cpp")
target_include_directories(fns_candy_style_transfer_cpp PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PNG_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  )
target_link_libraries(fns_candy_style_transfer_cpp PRIVATE
  onnxruntime
  ${CUDA_LIBRARIES}
  ${PNG_LIBRARIES})
if(PNG_LIBDIR)
  target_link_directories(fns_candy_style_transfer_cpp PRIVATE
    ${PNG_LIBDIR})
endif()
