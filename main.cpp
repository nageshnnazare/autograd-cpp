#include <iostream>

#include <tensor.h>

int main (int argc, char *argv[]) {
  auto w = AG::Tensor<float>::zeros({3, 5}, /*requires_grad=*/true);
  w->print("w");

  auto b = AG::Tensor<float>::zeros({3}, /*requires_grad=*/false);
  b->print("b");

  return 0;
}
