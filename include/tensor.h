#ifndef _AG_TENSOR_H_
#define _AG_TENSOR_H_

/**
 * @file tensor.h
 * @brief Multidimensional Tensor, TensorView, and Autograd engine core declarations.
 *
 * Defines the core Tensor template class with multi-dimensional indexing,
 * gradient tracking, factory methods (zeros, ones, rand, full, from), and
 * topological backward graph traversal.
 */

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace AG {

/**
 * @brief Represents the dimensional extents of a Tensor (e.g., {batch, channels, height, width}).
 */
using Shape = std::vector<std::size_t>;

/**
 * @brief Lightweight proxy view for chained multidimensional subscript indexing (e.g., `t[i][j]`).
 *
 * Each bracket operation steps forward through the dimension hierarchy, tracking
 * cumulative linear memory offset until the final scalar element is reached.
 *
 * @tparam T Element data type (e.g. float, double).
 */
template <typename T>
struct TensorView {
  T* base;             ///< Pointer to underlying flat storage buffer.
  const Shape* shape;  ///< Non-owning pointer to the tensor shape.
  int dim;             ///< Current dimension index being resolved.
  std::size_t offset;  ///< Current flat linear offset into base buffer.

  /**
   * @brief Calculates the linear memory stride of the current dimension.
   * @return Number of elements to step per unit in this dimension.
   */
  std::size_t stride() const {
    std::size_t s = 1;
    for (int d = dim + 1; d < static_cast<int>(shape->size()); ++d) s *= (*shape)[d];
    return s;
  }

  /**
   * @brief Advances one dimension deeper in the tensor view.
   * @param i Index along the current dimension.
   * @return A new TensorView narrowed to the next dimension.
   */
  TensorView operator[](std::size_t i) { return {base, shape, dim + 1, offset + i * stride()}; }

  /**
   * @brief Implicit conversion to scalar reference for read/write access.
   */
  operator T&() { return base[offset]; }

  /**
   * @brief Implicit conversion to const scalar reference for read access.
   */
  operator const T&() const { return base[offset]; }

  /**
   * @brief Assigns a scalar value to the element at the current linear offset.
   * @param val Scalar value to store.
   * @return Reference to this TensorView.
   */
  TensorView& operator=(const T& val) {
    base[offset] = val;
    return *this;
  }
};

/**
 * @brief Computes the total number of scalar elements represented by a Shape.
 * @param shape Tensor dimensions.
 * @return Total element count (product of all dimensions, or 1 for scalar 0D shape).
 */
std::size_t numel(const Shape& shape);

/**
 * @brief Converts multi-dimensional coordinates into a row-major linear flat index.
 * @param shape Shape of the tensor.
 * @param coords Multi-dimensional coordinate vector matching shape rank.
 * @return Flat linear memory index.
 */
std::size_t flatid(const Shape& shape, const std::vector<std::size_t>& coords);

/**
 * @brief N-dimensional dense array container with autograd backpropagation support.
 *
 * Supports multi-dimensional subscripting, initialization factories, gradient storage,
 * and computation graph dependency recording for automatic differentiation.
 *
 * @tparam T Underlying scalar type (e.g. float, double, int).
 */
template <typename T>
struct Tensor {
  Shape shape;          ///< Dimensions of the tensor.
  std::vector<T> data;  ///< Contiguous row-major storage for tensor values.
  std::string op_name;  ///< Name of operation that created this tensor (for graph visualization).

  bool requires_grad = false;  ///< Flag indicating if gradients should be tracked.
  std::vector<float> grad;     ///< Accumulated gradients with respect to this tensor.
  std::vector<std::shared_ptr<Tensor>> parents;  ///< Predecessor nodes in the computation graph.
  std::function<void()> backward_fn;             ///< Backward closure for gradient accumulation.

  /**
   * @brief Multi-dimensional indexing operator (mutable).
   * @param i Index along the outermost dimension.
   * @return A TensorView resolving index `i`.
   */
  TensorView<T> operator[](std::size_t i);

  /**
   * @brief Multi-dimensional indexing operator (const).
   * @param i Index along the outermost dimension.
   * @return A const TensorView resolving index `i`.
   */
  const TensorView<const T> operator[](std::size_t i) const;

  /**
   * @brief Creates a tensor filled with a constant scalar value.
   * @param shape Target dimensions.
   * @param fill_value Constant value to fill data buffer with.
   * @param requires_grad Whether gradient tracking is enabled.
   */
  static Tensor<T> full(Shape shape, T fill_value, bool requires_grad = false);

  /**
   * @brief Creates a tensor filled with zeros.
   * @param shape Target dimensions.
   * @param requires_grad Whether gradient tracking is enabled.
   */
  static Tensor<T> zeros(Shape shape, bool requires_grad = false);

  /**
   * @brief Creates a tensor filled with ones.
   * @param shape Target dimensions.
   * @param requires_grad Whether gradient tracking is enabled.
   */
  static Tensor<T> ones(Shape shape, bool requires_grad = false);

  /**
   * @brief Creates a tensor filled with uniform random values in [0.0, 1.0).
   * @param shape Target dimensions.
   * @param requires_grad Whether gradient tracking is enabled.
   */
  static Tensor<T> rand(Shape shape, bool requires_grad = false);

  /**
   * @brief Constructs a tensor from an existing shape and data buffer.
   * @param shape Target dimensions.
   * @param data Storage vector whose size must equal numel(shape).
   * @param requires_grad Whether gradient tracking is enabled.
   * @throws std::invalid_argument if data.size() != numel(shape).
   */
  static Tensor<T> from(Shape shape, std::vector<T> data, bool requires_grad = false);

  /**
   * @brief Resets all accumulated gradient values to zero.
   */
  void zero_grad();

  /**
   * @brief Computes gradients of the computation graph using reverse-mode autodiff.
   *
   * Traverses backward through parent dependencies via topological sort, invoking
   * each node's `backward_fn` closure to accumulate gradients.
   */
  void backward();

  /**
   * @brief Formats and prints the tensor shape and nested multidimensional contents to stdout.
   * @param name Optional label to prepend to the output.
   */
  void print(const char* name = "") const;

 private:
  /**
   * @brief Recursive helper to print formatted multi-dimensional bracketed values.
   */
  void print_dim(int dim, std::size_t offset, const char* base_indent) const;
};

}  // namespace AG

#endif  // !_AG_TENSOR_H_
