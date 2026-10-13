// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/SparseCore>

#include <memory>

struct ReallocationScalar {
  struct ConstructionFailure {};

  static int live_objects;
  static int live_resources;
  static int defaults_before_throw;
  static int moves_before_throw;

  std::unique_ptr<int> value;

  ReallocationScalar() {
#ifdef EIGEN_EXCEPTIONS
    if (defaults_before_throw == 0) throw ConstructionFailure{};
#endif
    if (defaults_before_throw > 0) --defaults_before_throw;
    value.reset(new int(0));
    ++live_objects;
    ++live_resources;
  }

  ReallocationScalar(ReallocationScalar&& other) {
#ifdef EIGEN_EXCEPTIONS
    if (moves_before_throw == 0) throw ConstructionFailure{};
#endif
    if (moves_before_throw > 0) --moves_before_throw;
    value = std::move(other.value);
    ++live_objects;
  }

  ~ReallocationScalar() {
    if (value) --live_resources;
    --live_objects;
  }

  static void reset_failures() {
    defaults_before_throw = -1;
    moves_before_throw = -1;
  }
};

int ReallocationScalar::live_objects = 0;
int ReallocationScalar::live_resources = 0;
int ReallocationScalar::defaults_before_throw = -1;
int ReallocationScalar::moves_before_throw = -1;

template <bool Align>
void reallocation_success() {
  using Scalar = ReallocationScalar;
  Scalar::reset_failures();
  Scalar* values = internal::conditional_aligned_new<Scalar, Align>(3);
  for (int i = 0; i < 3; ++i) *values[i].value = i + 1;
  for (int size : {5, 5, 2, 0}) {
    const int old_size = Scalar::live_objects;
    values = internal::conditional_aligned_realloc_new<Scalar, Align>(values, size, old_size);
    VERIFY_IS_EQUAL(Scalar::live_objects, size);
    VERIFY_IS_EQUAL(Scalar::live_resources, size);
    for (int i = 0; i < size; ++i) VERIFY_IS_EQUAL(*values[i].value, i < 3 ? i + 1 : 0);
  }
  internal::conditional_aligned_delete<Scalar, Align>(values, 0);
}

#ifdef EIGEN_EXCEPTIONS
template <bool Align>
void reallocation_failure(bool fail_move, int successful_constructions, int old_size) {
  using Scalar = ReallocationScalar;
  Scalar::reset_failures();
  Scalar* values = internal::conditional_aligned_new<Scalar, Align>(old_size);
  for (int i = 0; i < old_size; ++i) *values[i].value = i + 1;
  if (fail_move)
    Scalar::moves_before_throw = successful_constructions;
  else
    Scalar::defaults_before_throw = successful_constructions;

  bool caught = false;
  try {
    values = internal::conditional_aligned_realloc_new<Scalar, Align>(values, old_size + 2, old_size);
  } catch (const Scalar::ConstructionFailure&) {
    caught = true;
  }
  Scalar::reset_failures();
  VERIFY(caught);
  VERIFY_IS_EQUAL(Scalar::live_objects, old_size);
  // Only a failing move can leave the original elements moved-from.
  VERIFY_IS_EQUAL(Scalar::live_resources, fail_move ? old_size - successful_constructions : old_size);
  if (!fail_move) {
    for (int i = 0; i < old_size; ++i) VERIFY_IS_EQUAL(*values[i].value, i + 1);
  }
  internal::conditional_aligned_delete<Scalar, Align>(values, old_size);
  VERIFY_IS_EQUAL(Scalar::live_objects, 0);
  VERIFY_IS_EQUAL(Scalar::live_resources, 0);
}

void sparse_reserve_failure() {
  using Scalar = ReallocationScalar;
  for (int successful_defaults : {0, 1}) {
    Scalar::reset_failures();
    {
      SparseVector<Scalar> vector(10);
      vector.reserve(3);
      VERIFY_IS_EQUAL(Scalar::live_objects, 3);
      Scalar::defaults_before_throw = successful_defaults;
      bool caught = false;
      try {
        vector.reserve(5);
      } catch (const Scalar::ConstructionFailure&) {
        caught = true;
      }
      Scalar::reset_failures();
      VERIFY(caught);
      VERIFY_IS_EQUAL(vector.size(), 10);
      VERIFY_IS_EQUAL(vector.nonZeros(), 0);
      VERIFY_IS_EQUAL(Scalar::live_objects, 3);
      VERIFY_IS_EQUAL(Scalar::live_resources, 3);
    }
    VERIFY_IS_EQUAL(Scalar::live_objects, 0);
    VERIFY_IS_EQUAL(Scalar::live_resources, 0);
  }
}
#endif

EIGEN_DECLARE_TEST(memory_reallocation) {
  CALL_SUBTEST(reallocation_success<false>());
  CALL_SUBTEST(reallocation_success<true>());
#ifdef EIGEN_EXCEPTIONS
  CALL_SUBTEST(sparse_reserve_failure());
  for (int successful_constructions : {0, 1}) {
    CALL_SUBTEST(reallocation_failure<false>(false, successful_constructions, 3));
    CALL_SUBTEST(reallocation_failure<true>(false, successful_constructions, 3));
    CALL_SUBTEST(reallocation_failure<false>(true, successful_constructions, 3));
    CALL_SUBTEST(reallocation_failure<true>(true, successful_constructions, 3));
    CALL_SUBTEST(reallocation_failure<false>(false, successful_constructions, 0));
    CALL_SUBTEST(reallocation_failure<true>(false, successful_constructions, 0));
  }
#endif
}
