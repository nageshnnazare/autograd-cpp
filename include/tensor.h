#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <vector>
#include <numeric>

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
    for (int d = dim + 1; d < (int)shape->size(); ++d)
      s *= (*shape)[d];
    return s;
  }

  // Advance one dimension
  TensorView operator[](std::size_t i) {
    return { base, shape, dim + 1, offset + i * stride() };
  }

  // Access
  operator T&() {
    return base[offset];
  }

  operator const T&() const {
    return base[offset];
  }
};

std::size_t numel(const Shape& shape) {
  return std::accumulate(shape.begin(),
                         shape.end(),
                         static_cast<std::size_t>(1),
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
    for (std::size_t d = 1; d < shape.size(); ++d)
      stride *= shape[d];
    return { data.data(), &shape, 0, i * stride };
  }

  const TensorView<const T> operator[](std::size_t i) const {
    std::size_t stride = 1;
    for (std::size_t d = 1; d < shape.size(); ++d) 
      stride *= shape[d];
    return { data.data(), &shape, 0, i * stride };
  }

  static std::shared_ptr<Tensor<T>> zeros(Shape shape,
                                          bool requires_grad = false) {
    auto t = std::make_shared<Tensor>();
    t->shape = std::move(shape);
    T val = 0;
    t->data.assign(numel(t->shape), val);
    t->requires_grad = requires_grad;
    return t;
  }
  // static std::shared_ptr<Tensor>  ones(Shape shape,
  //                                      bool requires_grad = false);
  // static std::shared_ptr<Tensor>  rand(Shape shape,
  //                                      bool requires_grad = false);
  // static std::shared_ptr<Tensor>  full(Shape shape,
  //                                      bool requires_grad = false);
  // static std::shared_ptr<Tensor>  from(Shape shape,
  //                                      std::vector<float> data,
  //                                      bool requires_grad = false);

  void print(const char *name = "") const {
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
    for (int d = dim + 1; d < (int)shape.size(); ++d) 
      stride *= shape[d];

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

#endif // !_TENSOR_H_
