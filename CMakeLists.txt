cmake_minimum_required(VERSION 3.20)
project(autograd_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(autograd
  src/main.cpp
)

target_include_directories(autograd
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Keep the binary next to the build dir root
set_target_properties(autograd 
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

