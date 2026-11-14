#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// Usage:
//   auto w = AG::Tensor::zeros<float>({3, 5}, /*requires_grad=*/true);

namespace AG {

using Shape = std::vector<std::size_t>;

template <typename T>
struct TensorView {
  T* base;
  const Shape* shape;
  int dim;
  std::size_t offset;

  // Stride of the current dimension
  std::size_t stride() const {
    std::size_t s = 1;
    for (int d = dim + 1; d < (int)shape->size(); ++d) s *= (*shape)[d];
    return s;
  }

  // Advance one dimension
  TensorView operator[](std::size_t i) { return {base, shape, dim + 1, offset + i * stride()}; }

  // Access
  operator T&() { return base[offset]; }

  operator const T&() const { return base[offset]; }

  // assign
  TensorView& operator=(const T& val) {
    base[offset] = val;
    return *this;
  }
};

std::size_t numel(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1),
                         std::multiplies<>());
}

template <typename T>
struct Tensor {
  Shape shape;
  std::vector<T> data;
  std::vector<float> grad;
  bool requires_grad = false;

  TensorView<T> operator[](std::size_t i) {
    std::size_t stride = 1;
    for (std::size_t d = 1; d < shape.size(); ++d) stride *= shape[d];
    return {data.data(), &shape, 1, i * stride};
  }

  const TensorView<const T> operator[](std::size_t i) const {
    std::size_t stride = 1;
    for (std::size_t d = 1; d < shape.size(); ++d) stride *= shape[d];
    return {data.data(), &shape, 1, i * stride};
  }

  static Tensor<T> full(Shape shape, T fill_value, bool requires_grad = false) {
    Tensor<T> t;
    t.shape = std::move(shape);
    t.data.assign(numel(t.shape), fill_value);
    t.requires_grad = requires_grad;
    return t;
  }

  static Tensor<T> zeros(Shape shape, bool requires_grad = false) {
    return full(shape, T(0), requires_grad);
  }

  static Tensor<T> ones(Shape shape, bool requires_grad = false) {
    return full(shape, T(1), requires_grad);
  }

  static Tensor<T> rand(Shape shape, bool requires_grad = false) {
    Tensor<T> t;
    t.shape = std::move(shape);
    t.data.resize(numel(t.shape));
    t.requires_grad = requires_grad;

    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (auto& v : t.data) v = static_cast<T>(dist(rng));

    return t;
  }

  static Tensor<T> from(Shape shape, std::vector<T> data, bool requires_grad = false) {
    const std::size_t n = numel(shape);
    if (data.size() != n)
      throw std::invalid_argument("AG::Tensor::from — data size (" + std::to_string(data.size()) +
                                  ") does not match shape numel (" + std::to_string(n) + ")");

    Tensor<T> t;
    t.shape = std::move(shape);
    t.data = std::move(data);
    t.requires_grad = requires_grad;
    return t;
  }

  void print(const char* name = "") const {
    size_t indent = strlen(name) + 1;
    std::string pad(indent, ' ');

    std::printf("%s shape=[", name);
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (i) std::printf(",");
      std::printf("%zu", shape[i]);
    }
    std::printf("]\n");
    std::printf("%sdata=", pad.c_str());
    print_dim(0, 0, pad.c_str());
    std::printf("\n");
  }

 private:
  void print_dim(int dim, std::size_t offset, const char* base_indent) const {
    if (dim == static_cast<int>(shape.size())) {
      std::printf("%.6g", data[offset]);
      return;
    }

    std::printf("[");

    std::size_t stride = 1;
    for (int d = dim + 1; d < (int)shape.size(); ++d) stride *= shape[d];

    for (std::size_t i = 0; i < shape[dim]; ++i) {
      if (i) std::printf(",");
      if (dim + 1 < static_cast<int>(shape.size())) {
        std::printf("\n");
        std::printf("%s", base_indent);
        for (int d = 0; d <= dim; ++d) std::printf("  ");
      }
      print_dim(dim + 1, offset + i * stride, base_indent);
    }

    if (dim + 1 < (int)shape.size() && shape[dim] > 0) {
      std::printf("\n");
      std::printf("%s", base_indent);
      for (int d = 0; d < dim; ++d) std::printf("  ");
    }

    std::printf("]");
  }
};

}  // namespace AG

#endif  // !_TENSOR_H_
