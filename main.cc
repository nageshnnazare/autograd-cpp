/**
 * @file main.cc
 * @brief Demonstration executable for AG::Tensor construction and operations.
 */

#include <tensor.h>

#include <iostream>

int main(int argc, char* argv[]) {
  // Construct a 2D weight matrix initialized to zeros with gradient tracking enabled
  auto w = AG::Tensor<float>::zeros({3, 5}, /*requires_grad=*/true);
  w.print("w");

  // Construct a 1D bias vector initialized to zeros without gradient tracking
  auto b = AG::Tensor<float>::zeros({3}, /*requires_grad=*/false);
  b.print("b");

  // Demonstrate multi-dimensional subscript indexing
  std::cout << "w[0][1] = " << w[0][1] << std::endl;

  return 0;
}
