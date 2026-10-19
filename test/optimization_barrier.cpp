// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>

enum class BarrierEnum : unsigned long long { Value = 0xfedcba9876543210ull };
struct BarrierRecord {
  unsigned long long value;
};
union BarrierUnion {
  unsigned long long value;
  double real;
};

template <typename T>
void check_barrier_value(T value) {
  const T expected = value;
  EIGEN_OPTIMIZATION_BARRIER(value);
  VERIFY(value == expected);
  internal::pbarrier(value);
  VERIFY(value == expected);
}

template <typename Scalar>
EIGEN_DONT_INLINE Scalar rounded_by_barrier(Scalar x) {
  const Scalar bias = numext::ldexp(Scalar(1), NumTraits<Scalar>::digits() - 1);
  Scalar rounded = x + bias;
  EIGEN_OPTIMIZATION_BARRIER(rounded);
  return rounded - bias;
}

EIGEN_DECLARE_TEST(optimization_barrier) {
  check_barrier_value(false);
  check_barrier_value(static_cast<signed char>(-17));
  check_barrier_value(static_cast<unsigned short>(65521));
  check_barrier_value(-1234567);
  check_barrier_value(0xfedcba9876543210ull);
  check_barrier_value(BarrierEnum::Value);
  check_barrier_value(nullptr);
  check_barrier_value(1.25f);
  check_barrier_value(1.25);
  check_barrier_value(1.25L);
  int pointee = 7;
  check_barrier_value(&pointee);
  volatile decltype(nullptr) null_value = nullptr;
  EIGEN_OPTIMIZATION_BARRIER(null_value);
  VERIFY(null_value == nullptr);
  volatile double real_value = 1.25;
  EIGEN_OPTIMIZATION_BARRIER(real_value);
  VERIFY(real_value == 1.25);

  BarrierRecord record{0xfedcba9876543210ull};
  EIGEN_OPTIMIZATION_BARRIER(record);
  VERIFY_IS_EQUAL(record.value, 0xfedcba9876543210ull);
  BarrierUnion object;
  object.value = 0xfedcba9876543210ull;
  EIGEN_OPTIMIZATION_BARRIER(object);
  VERIFY_IS_EQUAL(object.value, 0xfedcba9876543210ull);
  int array[] = {1, -2, 3, -4};
  EIGEN_OPTIMIZATION_BARRIER(array);
  for (int i = 0; i < 4; ++i) VERIFY_IS_EQUAL(array[i], (i % 2 ? -1 : 1) * (i + 1));

#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(SYCL_DEVICE_ONLY) && \
    (EIGEN_COMP_GNUC || (defined(__clang__) && defined(_MSC_VER)))
  VERIFY_IS_EQUAL(rounded_by_barrier(1.25f), 1.0f);
  VERIFY_IS_EQUAL(rounded_by_barrier(1.25), 1.0);
#endif

#if !defined(EIGEN_GPU_COMPILE_PHASE) &&                                       \
    ((EIGEN_ARCH_ARM_OR_ARM64 && defined(__ARM_FP)) ||                         \
     (EIGEN_ARCH_i386_OR_x86_64 && defined(__SSE2__) && !EIGEN_COMP_NVHPC)) && \
    (EIGEN_COMP_GNUC_STRICT || EIGEN_COMP_CLANG)
  using internal::optimization_barrier_in_register;
#if !EIGEN_ARCH_ARM_OR_ARM64 || (__ARM_FP & 4)
  STATIC_CHECK(optimization_barrier_in_register<float>::value);
#endif
#if !EIGEN_ARCH_ARM_OR_ARM64 || (__ARM_FP & 8)
  STATIC_CHECK(optimization_barrier_in_register<double>::value);
#endif
  STATIC_CHECK(!optimization_barrier_in_register<long double>::value);
  STATIC_CHECK(!optimization_barrier_in_register<int>::value);
  STATIC_CHECK(!optimization_barrier_in_register<int*>::value);
  STATIC_CHECK(!optimization_barrier_in_register<decltype(nullptr)>::value);
  STATIC_CHECK(!optimization_barrier_in_register<const volatile decltype(nullptr)>::value);
  STATIC_CHECK(!optimization_barrier_in_register<BarrierEnum>::value);
  STATIC_CHECK(!optimization_barrier_in_register<BarrierRecord>::value);
  STATIC_CHECK(!optimization_barrier_in_register<BarrierUnion>::value);
  STATIC_CHECK(!optimization_barrier_in_register<Eigen::half>::value);
  STATIC_CHECK(!optimization_barrier_in_register<Eigen::bfloat16>::value);
  STATIC_CHECK(!optimization_barrier_in_register<std::complex<double>>::value);
  STATIC_CHECK(!optimization_barrier_in_register<int[4]>::value);
#if EIGEN_ARCH_i386_OR_x86_64 || defined(__ARM_NEON)
  using BarrierVector = float __attribute__((vector_size(16)));
  using WideBarrierVector = float __attribute__((vector_size(128)));
  STATIC_CHECK(optimization_barrier_in_register<BarrierVector>::value);
  STATIC_CHECK(!optimization_barrier_in_register<WideBarrierVector>::value);
#endif
#endif
}
