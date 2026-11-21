/**
 * @file test_tensor.cc
 * @brief Zero-dependency unit test suite and runner for AG::Tensor.
 *
 * Provides a minimal lightweight test harness that registers named test cases,
 * executes them, and reports per-test PASS/FAIL status. Can run all tests or
 * target a specific test by name when invoked from CTest.
 */

#include <tensor.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------

/**
 * @brief Represents an individual test case with an identifier and runner callable.
 */
struct TestCase {
  const char* name;
  std::function<void()> fn;
};

/// Global registry of registered test cases.
static std::vector<TestCase> g_tests;

/**
 * @def TEST(name)
 * @brief Registers a test case with the global test harness.
 */
#define TEST(name)                           \
  static void test_##name();                 \
  static bool reg_##name = [] {              \
    g_tests.push_back({#name, test_##name}); \
    return true;                             \
  }();                                       \
  static void test_##name()

/**
 * @def EXPECT(cond)
 * @brief Asserts that a condition is true, throwing an exception on failure.
 */
#define EXPECT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      throw std::runtime_error("assertion failed");                           \
    }                                                                         \
  } while (0)

/**
 * @def EXPECT_NEAR(a, b, tol)
 * @brief Asserts that two floating-point values are within an absolute tolerance.
 */
#define EXPECT_NEAR(a, b, tol) EXPECT(std::fabs((a) - (b)) <= (tol))

// ---------------------------------------------------------------------------
// Test Cases
// ---------------------------------------------------------------------------

/// Verify shape and total element allocation of zeros factory.
TEST(zeros_shape) {
  auto t = AG::Tensor<float>::zeros({3, 5});
  EXPECT(t.shape.size() == 2);
  EXPECT(t.shape[0] == 3);
  EXPECT(t.shape[1] == 5);
  EXPECT(t.data.size() == 15);
}

/// Verify that zeros tensor elements are initialized to 0.
TEST(zeros_values) {
  auto t = AG::Tensor<float>::zeros({2, 4});
  for (auto v : t.data) EXPECT_NEAR(v, 0.f, 1e-9f);
}

/// Verify that ones tensor elements are initialized to 1.
TEST(ones_values) {
  auto t = AG::Tensor<float>::ones({2, 3});
  for (auto v : t.data) EXPECT_NEAR(v, 1.f, 1e-9f);
}

/// Verify that full tensor elements match the requested fill value.
TEST(full_values) {
  auto t = AG::Tensor<float>::full({3, 3}, 7.f);
  for (auto v : t.data) EXPECT_NEAR(v, 7.f, 1e-9f);
}

/// Verify that rand generates elements in the range [0.0, 1.0).
TEST(rand_range) {
  auto t = AG::Tensor<float>::rand({10, 10});
  for (auto v : t.data) {
    EXPECT(v >= 0.f);
    EXPECT(v < 1.f);
  }
}

/// Verify valid initialization from existing data buffer.
TEST(from_happy) {
  auto t = AG::Tensor<float>::from({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT(t.shape[0] == 2 && t.shape[1] == 3);
  EXPECT_NEAR(t.data[0], 1.f, 1e-9f);
  EXPECT_NEAR(t.data[5], 6.f, 1e-9f);
}

/// Verify that mismatched data size and shape numel throws std::invalid_argument.
TEST(from_mismatch_throws) {
  bool threw = false;
  try {
    AG::Tensor<float>::from({2, 3}, {1, 2, 3});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);
}

/// Verify multi-dimensional subscript read access.
TEST(indexing_read) {
  auto t = AG::Tensor<float>::from({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT_NEAR(static_cast<float>(t[0][0]), 1.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>(t[0][2]), 3.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>(t[1][0]), 4.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>(t[1][2]), 6.f, 1e-9f);
}

/// Verify multi-dimensional subscript write access and isolation.
TEST(indexing_write) {
  auto t = AG::Tensor<float>::zeros({2, 3});
  t[1][2] = 42.f;
  EXPECT_NEAR(static_cast<float>(t[1][2]), 42.f, 1e-9f);
  // neighbouring elements must be untouched
  EXPECT_NEAR(static_cast<float>(t[1][1]), 0.f, 1e-9f);
}

/// Verify requires_grad initialization flag.
TEST(requires_grad_flag) {
  auto a = AG::Tensor<float>::zeros({2}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::zeros({2}, /*requires_grad=*/false);
  EXPECT(a.requires_grad == true);
  EXPECT(b.requires_grad == false);
}

/// Verify numel calculation and row-major flat coordinate indexing.
TEST(size_and_elements) {
  AG::Shape shape{2, 3};
  auto t = AG::Tensor<float>::from(shape, {1, 2, 3, 4, 5, 6});
  EXPECT(AG::numel(shape) == 6);
  EXPECT(AG::flatid(shape, {1, 1}) == 4);
  EXPECT(t.data[AG::flatid(shape, {1, 2})] == 6);
}

/// Verify gradient initialization and backward pass execution on leaf tensor.
TEST(backward_leaf) {
  auto t = AG::Tensor<float>::zeros({2, 3}, /*requires_grad=*/true);
  t.backward();
  EXPECT(t.grad.size() == 6);
  for (auto g : t.grad) EXPECT_NEAR(g, 1.f, 1e-9f);
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------
/**
 * @brief Test runner entry point.
 *
 * Supports three invocation modes:
 * - No arguments: Runs all registered tests in sequence and summarizes results.
 * - `--list`: Prints all registered test names (one per line).
 * - `<test_name>`: Runs only the specified test case, exiting 0 on PASS, 1 on FAIL.
 */
int main(int argc, char* argv[]) {
  // If an argument is provided, support listing or running a specific test
  if (argc > 1) {
    std::string arg = argv[1];

    if (arg == "--list") {
      for (const auto& tc : g_tests) {
        std::printf("%s\n", tc.name);
      }
      return 0;
    }

    for (const auto& tc : g_tests) {
      if (arg == tc.name) {
        try {
          tc.fn();
          std::printf("  PASS  %s\n", tc.name);
          return 0;
        } catch (...) {
          std::printf("  FAIL  %s\n", tc.name);
          return 1;
        }
      }
    }

    std::fprintf(stderr, "Unknown test: %s\n", arg.c_str());
    return 1;
  }

  // Run all tests
  int passed = 0, failed = 0;
  for (auto& tc : g_tests) {
    try {
      tc.fn();
      std::printf("  PASS  %s\n", tc.name);
      ++passed;
    } catch (...) {
      std::printf("  FAIL  %s\n", tc.name);
      ++failed;
    }
  }
  std::printf("\n%d / %d tests passed\n", passed, passed + failed);
  return failed ? 1 : 0;
}
