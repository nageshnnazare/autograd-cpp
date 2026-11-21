// Minimal zero-dependency test runner for AG::Tensor.
// Each TEST() registers a named case; the binary exits non-zero on any failure.

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
struct TestCase {
  const char* name;
  std::function<void()> fn;
};

static std::vector<TestCase> g_tests;

#define TEST(name)                           \
  static void test_##name();                 \
  static bool reg_##name = [] {              \
    g_tests.push_back({#name, test_##name}); \
    return true;                             \
  }();                                       \
  static void test_##name()

#define EXPECT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      throw std::runtime_error("assertion failed");                           \
    }                                                                         \
  } while (0)

#define EXPECT_NEAR(a, b, tol) EXPECT(std::fabs((a) - (b)) <= (tol))

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(zeros_shape) {
  auto t = AG::Tensor<float>::zeros({3, 5});
  EXPECT(t.shape.size() == 2);
  EXPECT(t.shape[0] == 3);
  EXPECT(t.shape[1] == 5);
  EXPECT(t.data.size() == 15);
}

TEST(zeros_values) {
  auto t = AG::Tensor<float>::zeros({2, 4});
  for (auto v : t.data) EXPECT_NEAR(v, 0.f, 1e-9f);
}

TEST(ones_values) {
  auto t = AG::Tensor<float>::ones({2, 3});
  for (auto v : t.data) EXPECT_NEAR(v, 1.f, 1e-9f);
}

TEST(full_values) {
  auto t = AG::Tensor<float>::full({3, 3}, 7.f);
  for (auto v : t.data) EXPECT_NEAR(v, 7.f, 1e-9f);
}

TEST(rand_range) {
  auto t = AG::Tensor<float>::rand({10, 10});
  for (auto v : t.data) {
    EXPECT(v >= 0.f);
    EXPECT(v < 1.f);
  }
}

TEST(from_happy) {
  auto t = AG::Tensor<float>::from({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT(t.shape[0] == 2 && t.shape[1] == 3);
  EXPECT_NEAR(t.data[0], 1.f, 1e-9f);
  EXPECT_NEAR(t.data[5], 6.f, 1e-9f);
}

TEST(from_mismatch_throws) {
  bool threw = false;
  try {
    AG::Tensor<float>::from({2, 3}, {1, 2, 3});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);
}

TEST(indexing_read) {
  auto t = AG::Tensor<float>::from({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT_NEAR(static_cast<float>(t[0][0]), 1.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>(t[0][2]), 3.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>(t[1][0]), 4.f, 1e-9f);
  EXPECT_NEAR(static_cast<float>(t[1][2]), 6.f, 1e-9f);
}

TEST(indexing_write) {
  auto t = AG::Tensor<float>::zeros({2, 3});
  t[1][2] = 42.f;
  EXPECT_NEAR(static_cast<float>(t[1][2]), 42.f, 1e-9f);
  // neighbouring elements must be untouched
  EXPECT_NEAR(static_cast<float>(t[1][1]), 0.f, 1e-9f);
}

TEST(requires_grad_flag) {
  auto a = AG::Tensor<float>::zeros({2}, /*requires_grad=*/true);
  auto b = AG::Tensor<float>::zeros({2}, /*requires_grad=*/false);
  EXPECT(a.requires_grad == true);
  EXPECT(b.requires_grad == false);
}

TEST(size_and_elements) {
  AG::Shape shape{2, 3};
  auto t = AG::Tensor<float>::from(shape, {1, 2, 3, 4, 5, 6});
  EXPECT(AG::numel(shape) == 6);
  EXPECT(AG::flatid(shape, {1, 1}) == 4);
  EXPECT(t.data[AG::flatid(shape, {1, 2})] == 6);
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------
int main() {
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
