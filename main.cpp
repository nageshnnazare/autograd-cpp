#include <tensor.h>

#include <iostream>

int main(int argc, char* argv[]) {
  auto w = AG::Tensor<float>::zeros({3, 5}, /*requires_grad=*/true);
  w.print("w");

  auto b = AG::Tensor<float>::zeros({3}, /*requires_grad=*/false);
  b.print("b");

  std::cout << w[0][1] << std::endl;

  return 0;
}
