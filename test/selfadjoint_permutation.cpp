// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <Eigen/Core>
#include <algorithm>
#include <cstdint>

template <unsigned int UpLo, typename MatrixType, typename PermutationDerived>
void check_selfadjoint_permutation(MatrixType& matrix, const PermutationBase<PermutationDerived>& permutation) {
  using PlainObject = typename MatrixType::PlainObject;
  const PlainObject original = matrix;
  const Index size = matrix.rows();
  PlainObject full(size, size);
  for (Index j = 0; j < size; ++j) {
    for (Index i = 0; i < size; ++i) {
      full(Index(permutation.indices()[i]), Index(permutation.indices()[j])) =
          (UpLo == Lower ? i >= j : i <= j) ? original(i, j) : numext::conj(original(j, i));
    }
  }
  PlainObject expected = original;
  expected.template triangularView<UpLo>() = full;
  auto view = matrix.template selfadjointView<UpLo>();
  VERIFY(&view.applyPermutation(permutation) == &view);
  VERIFY_IS_EQUAL(matrix, expected);
  VERIFY_IS_EQUAL(view.toDenseMatrix(), full);
  view.applyPermutation(permutation.inverse());
  VERIFY_IS_EQUAL(matrix, original);
  view.applyPermutation(permutation.transpose()).applyPermutation(permutation);
  VERIFY_IS_EQUAL(matrix, original);
}

template <typename Scalar, int Order>
void selfadjoint_permutation_cycles() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  for (Index size = 0; size <= 5; ++size) {
    MatrixType matrix = MatrixType::Random(size, size);
    matrix.diagonal() = matrix.diagonal().real().template cast<Scalar>();
    PermutationMatrix<Dynamic> permutation(size);
    permutation.setIdentity();
    do {
      check_selfadjoint_permutation<Lower>(matrix, permutation);
      check_selfadjoint_permutation<Upper>(matrix, permutation);
    } while (size > 1 && std::next_permutation(permutation.indices().data(), permutation.indices().data() + size));
  }
}

template <typename Scalar, int Order>
void selfadjoint_permutation_views() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Index size = internal::random<Index>(2, 30);
  MatrixType storage = MatrixType::Random(2 * size + 3, 2 * size + 3);
  const MatrixType original = storage;
  auto block = storage.block(1, 2, size, size);
  PermutationMatrix<Dynamic, Dynamic, std::int64_t> permutation(size);
  randomPermutationVector(permutation.indices(), size);
  const Map<PermutationMatrix<Dynamic, Dynamic, std::int64_t>> mapped_permutation(permutation.indices().data(), size);
  check_selfadjoint_permutation<Lower>(block, mapped_permutation);
  check_selfadjoint_permutation<Upper>(block, permutation.indices().asPermutation());
  VERIFY_IS_EQUAL(storage, original);

  Map<MatrixType, Unaligned, Stride<Dynamic, Dynamic>> strided(storage.data(), size, size,
                                                               Stride<Dynamic, Dynamic>(2 * storage.outerStride(), 2));
  check_selfadjoint_permutation<Lower>(strided, permutation);
  check_selfadjoint_permutation<Upper>(strided, permutation);
  VERIFY_IS_EQUAL(storage, original);
}

template <typename Scalar, int Size>
void selfadjoint_permutation_fixed() {
  Matrix<Scalar, Size, Size> matrix = Matrix<Scalar, Size, Size>::Random();
  PermutationMatrix<Size> permutation;
  randomPermutationVector(permutation.indices(), Size);
  internal::set_is_malloc_allowed(false);
  check_selfadjoint_permutation<Lower>(matrix, permutation);
  check_selfadjoint_permutation<Upper>(matrix, permutation);
  internal::set_is_malloc_allowed(true);
}

void selfadjoint_permutation_preconditions() {
  MatrixXd matrix = MatrixXd::Random(3, 3);
  PermutationMatrix<Dynamic> permutation(2);
  permutation.setIdentity();
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Lower>().applyPermutation(permutation));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Upper>().applyPermutation(permutation.inverse()));
  matrix.resize(2, 3);
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Lower>().applyPermutation(permutation));
  VERIFY_RAISES_ASSERT(matrix.selfadjointView<Upper>().applyPermutation(permutation));
}

void selfadjoint_permutation_aliased_indices() {
  Matrix3i matrix;
  matrix << 1, 2, 0, 4, 5, 6, 7, 8, 9;
  const PermutationMatrix<3> permutation(matrix.row(0).transpose());
  Matrix3i expected = matrix;
  expected.selfadjointView<Upper>().applyPermutation(permutation);
  matrix.selfadjointView<Upper>().applyPermutation(matrix.row(0).transpose().asPermutation());
  VERIFY_IS_EQUAL(matrix, expected);
}

EIGEN_DECLARE_TEST(selfadjoint_permutation) {
  CALL_SUBTEST_1((selfadjoint_permutation_cycles<float, ColMajor>()));
  CALL_SUBTEST_1((selfadjoint_permutation_cycles<float, RowMajor>()));
  CALL_SUBTEST_2((selfadjoint_permutation_cycles<double, ColMajor>()));
  CALL_SUBTEST_2((selfadjoint_permutation_cycles<double, RowMajor>()));
  CALL_SUBTEST_3((selfadjoint_permutation_cycles<std::complex<float>, ColMajor>()));
  CALL_SUBTEST_3((selfadjoint_permutation_cycles<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_4((selfadjoint_permutation_cycles<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_4((selfadjoint_permutation_cycles<std::complex<double>, RowMajor>()));
  CALL_SUBTEST_5((selfadjoint_permutation_cycles<int, ColMajor>()));
  CALL_SUBTEST_5((selfadjoint_permutation_cycles<short, RowMajor>()));
  CALL_SUBTEST_5((selfadjoint_permutation_cycles<half, ColMajor>()));
  CALL_SUBTEST_5((selfadjoint_permutation_cycles<bfloat16, RowMajor>()));
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1((selfadjoint_permutation_views<float, ColMajor>()));
    CALL_SUBTEST_2((selfadjoint_permutation_views<double, RowMajor>()));
    CALL_SUBTEST_3((selfadjoint_permutation_views<std::complex<float>, RowMajor>()));
    CALL_SUBTEST_4((selfadjoint_permutation_views<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_1((selfadjoint_permutation_fixed<float, 0>()));
    CALL_SUBTEST_1((selfadjoint_permutation_fixed<float, 1>()));
    CALL_SUBTEST_2((selfadjoint_permutation_fixed<double, 4>()));
    CALL_SUBTEST_3((selfadjoint_permutation_fixed<std::complex<float>, 5>()));
  }
  CALL_SUBTEST_1(selfadjoint_permutation_preconditions());
  CALL_SUBTEST_5(selfadjoint_permutation_aliased_indices());
}
