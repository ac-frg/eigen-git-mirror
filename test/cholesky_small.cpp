// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Cholesky>

template <typename MatrixType, int UpLo>
void cholesky_small_positive(const MatrixType& matrix) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using VectorType = Matrix<Scalar, MatrixType::RowsAtCompileTime, 1>;
  const Index size = matrix.rows();
  MatrixType input = MatrixType::Constant(size, size, Scalar(NumTraits<RealScalar>::quiet_NaN()));
  input.template triangularView<UpLo>() = matrix;
  LLT<MatrixType, UpLo> llt(input);
  LDLT<MatrixType, UpLo> ldlt(input);
  VERIFY_IS_EQUAL(llt.info(), Success);
  VERIFY_IS_EQUAL(ldlt.info(), Success);
  VERIFY(ldlt.isPositive());

  // O(n*eps) backward error, including the reconstruction and residual products.
  const RealScalar reconstruction_bound = RealScalar(32 * size) * NumTraits<RealScalar>::epsilon() * matrix.norm();
  VERIFY((numext::isfinite)(reconstruction_bound));
  VERIFY((llt.reconstructedMatrix() - matrix).norm() <= reconstruction_bound);
  VERIFY((ldlt.reconstructedMatrix() - matrix).norm() <= reconstruction_bound);
  const VectorType rhs = VectorType::Random(size);
  for (int method = 0; method < 2; ++method) {
    VectorType solution(size);
    if (method == 0)
      solution = llt.solve(rhs);
    else
      solution = ldlt.solve(rhs);
    const RealScalar residual_bound =
        RealScalar(32 * size) * NumTraits<RealScalar>::epsilon() * (matrix.norm() * solution.norm() + rhs.norm());
    VERIFY((numext::isfinite)(residual_bound));
    VERIFY((matrix * solution - rhs).norm() <= residual_bound);
  }

  // Exercise the same update with both a runtime inner stride and in-place factor storage.
  using StrideType = Stride<Dynamic, Dynamic>;
  const Index inner = 2, outer = 2 * size + 3;
  const Index count = (size - 1) * (outer + inner) + 1;
  Matrix<Scalar, Dynamic, 1> buffer = Matrix<Scalar, Dynamic, 1>::Constant(count, Scalar(17));
  Map<MatrixType, 0, StrideType> mapped(buffer.data(), size, size, StrideType(outer, inner));
  using RefType = Ref<MatrixType, 0, StrideType>;
  mapped = input;
  LLT<RefType, UpLo> inplace_llt(mapped);
  VERIFY_IS_EQUAL(inplace_llt.info(), Success);
  const MatrixType llt_lower = inplace_llt.matrixL();
  VERIFY((llt_lower * llt_lower.adjoint() - matrix).norm() <= reconstruction_bound);
  mapped = input;
  LDLT<RefType, UpLo> inplace_ldlt(mapped);
  VERIFY_IS_EQUAL(inplace_ldlt.info(), Success);
  const MatrixType ldlt_lower = inplace_ldlt.matrixL();
  const MatrixType permuted = ldlt_lower * inplace_ldlt.vectorD().asDiagonal() * ldlt_lower.adjoint();
  const MatrixType reconstructed =
      inplace_ldlt.transpositionsP().transpose() * permuted * inplace_ldlt.transpositionsP();
  VERIFY((reconstructed - matrix).norm() <= reconstruction_bound);
  for (Index i = 0; i < count; ++i) {
    const Index offset = i % outer;
    if (offset >= inner * size || offset % inner != 0) VERIFY_IS_EQUAL(buffer(i), Scalar(17));
  }
}

template <typename MatrixType, int UpLo>
void cholesky_small_zero_pivot(Index size) {
  using Scalar = typename MatrixType::Scalar;
  MatrixType input = MatrixType::Zero(size, size);
  input.template topLeftCorner<3, 3>() << Scalar(4), Scalar(2), Scalar(2), Scalar(2), Scalar(1), Scalar(1), Scalar(2),
      Scalar(1), Scalar(1);
  LDLT<MatrixType, UpLo> factor(input);
  VERIFY_IS_EQUAL(factor.info(), Success);
  VERIFY(factor.isPositive());
  VERIFY_IS_CWISE_EQUAL(factor.reconstructedMatrix(), input);
  // At the second pivot A21 is nonzero before the update, but its Schur complement is exactly zero.
  VERIFY_IS_EQUAL(factor.vectorD()(1), Scalar(0));
  input = -input;
  factor.compute(input);
  VERIFY_IS_EQUAL(factor.info(), Success);
  VERIFY(factor.isNegative());
  VERIFY_IS_CWISE_EQUAL(factor.reconstructedMatrix(), input);

  input(1, 2) = input(2, 1) = Scalar(-2);
  factor.compute(input);
  VERIFY_IS_EQUAL(factor.info(), NumericalIssue);
  input.setZero();
  factor.compute(input);
  VERIFY_IS_EQUAL(factor.info(), Success);
  VERIFY(factor.isPositive() && factor.isNegative());
  input(0, 1) = input(1, 0) = Scalar(1);
  factor.compute(input);
  VERIFY_IS_EQUAL(factor.info(), NumericalIssue);
  LLT<MatrixType, UpLo> llt(input);
  VERIFY_IS_EQUAL(llt.info(), NumericalIssue);
}

template <typename Scalar, int Size, int Order>
void cholesky_small_update(Index size) {
  using MatrixType = Matrix<Scalar, Size, Size, Order>;
  using RealScalar = typename MatrixType::RealScalar;
  MatrixType lower = MatrixType::Zero(size, size);
  for (Index j = 0; j < size; ++j) {
    lower(j, j) = RealScalar(j + 1);
    for (Index i = j + 1; i < size; ++i) lower(i, j) = internal::random<Scalar>() / RealScalar(4);
  }
  const MatrixType matrix = lower * lower.adjoint();
  for (RealScalar scale : {RealScalar(1) / RealScalar(1024), RealScalar(1), RealScalar(1024)}) {
    const MatrixType scaled = matrix * scale;
    cholesky_small_positive<MatrixType, Lower>(scaled);
    cholesky_small_positive<MatrixType, Upper>(scaled);
  }
  if (size >= 3) {
    cholesky_small_zero_pivot<MatrixType, Lower>(size);
    cholesky_small_zero_pivot<MatrixType, Upper>(size);
  }
}

template <typename Scalar>
void cholesky_small_shapes() {
  cholesky_small_update<Scalar, 3, ColMajor>(3);
  cholesky_small_update<Scalar, 3, RowMajor>(3);
  cholesky_small_update<Scalar, 4, ColMajor>(4);
  cholesky_small_update<Scalar, 4, RowMajor>(4);
  for (Index size = 1; size <= 6; ++size) {
    cholesky_small_update<Scalar, Dynamic, ColMajor>(size);
    cholesky_small_update<Scalar, Dynamic, RowMajor>(size);
  }
}

EIGEN_DECLARE_TEST(cholesky_small) {
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    CALL_SUBTEST_1(cholesky_small_shapes<float>());
    CALL_SUBTEST_2(cholesky_small_shapes<double>());
    CALL_SUBTEST_3(cholesky_small_shapes<std::complex<float> >());
    CALL_SUBTEST_4(cholesky_small_shapes<std::complex<double> >());
  }
}
