// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

std::vector<Index> blas_singleton_strides(Index count) {
  std::vector<Index> strides{-1, 0, 1, count - 1, count, count + 3};
  EIGEN_IF_CONSTEXPR (std::numeric_limits<Index>::digits > std::numeric_limits<int>::digits) {
    Index large_stride = (std::numeric_limits<int>::max)();
    strides.push_back(++large_stride);
  }
  return strides;
}

void blas_stride_helpers() {
  constexpr Index count = 24;
  for (Index stride : blas_singleton_strides(count)) {
    VERIFY_IS_EQUAL(internal::blas_leading_dim<ColMajor>(count, Index(1), stride), count);
    VERIFY_IS_EQUAL(internal::blas_leading_dim<RowMajor>(Index(1), count, stride), count);
    VERIFY_IS_EQUAL(internal::blas_vector_increment(Index(1), stride), 1);
  }
  VERIFY_IS_EQUAL(internal::blas_leading_dim<ColMajor>(Index(0), Index(1), Index(0)), 1);
  VERIFY_IS_EQUAL(internal::blas_leading_dim<RowMajor>(Index(1), Index(0), Index(0)), 1);
  VERIFY_IS_EQUAL(internal::blas_leading_dim<ColMajor>(count, count, count + 3), count + 3);
  VERIFY_IS_EQUAL(internal::blas_vector_increment(count, Index(-2)), -2);
}

template <typename Scalar>
Scalar blas_stride_value(Index i) {
  return Scalar(i + 1);
}
template <>
std::complex<float> blas_stride_value<std::complex<float>>(Index i) {
  return {float(i + 1), float(i % 3 - 1)};
}
template <>
std::complex<double> blas_stride_value<std::complex<double>>(Index i) {
  return {double(i + 1), double(i % 3 - 1)};
}

template <typename Scalar, int Order>
void blas_singleton_stride() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  constexpr Index count = 24;
  const Index rows = Order == ColMajor ? count : 1;
  const Index cols = Order == ColMajor ? 1 : count;
  Scalar storage[count];
  for (Index i = 0; i < count; ++i) storage[i] = Scalar(i + 1);

  for (Index stride : blas_singleton_strides(count)) {
    // With one outer vector, even a zero or negative outer stride is unused.
    Map<const MatrixType, 0, OuterStride<>> matrix(storage, rows, cols, OuterStride<>(stride));
    const Scalar alpha = internal::random<Scalar>();
    const VectorType rhs = VectorType::Constant(cols, Scalar(2));
    VectorType expected(rows);
    for (Index i = 0; i < rows; ++i) {
      Scalar value = Scalar(0);
      for (Index j = 0; j < cols; ++j) value += matrix(i, j) * rhs(j);
      expected(i) = value;
    }
    VectorType result = matrix * rhs;
    VERIFY_IS_CWISE_EQUAL(result, expected);
    EIGEN_IF_CONSTEXPR (Order == RowMajor) {
      Scalar singleton_storage = Scalar(0);
      Map<VectorType, 0, InnerStride<>> singleton_result(&singleton_storage, 1, InnerStride<>(stride));
      singleton_result.noalias() = matrix * rhs;
      VERIFY_IS_EQUAL(singleton_storage, expected(0));
    }
    result.setOnes();
    result.noalias() += alpha * matrix * rhs;
    const RealScalar bound = RealScalar(8 * count) * NumTraits<RealScalar>::epsilon();
    VERIFY((result - (VectorType::Ones(rows) + alpha * expected)).norm() <= bound * (RealScalar(1) + expected.norm()));

    const VectorType transpose_rhs = VectorType::Constant(rows, Scalar(2));
    VectorType transpose_expected(cols);
    for (Index j = 0; j < cols; ++j) {
      Scalar value = Scalar(0);
      for (Index i = 0; i < rows; ++i) value += matrix(i, j) * transpose_rhs(i);
      transpose_expected(j) = value;
    }
    result = matrix.transpose() * transpose_rhs;
    VERIFY_IS_CWISE_EQUAL(result, transpose_expected);
  }
}

template <typename Scalar, int Order>
void blas_singleton_outer_product() {
  using ColMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RowMatrix = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  using DstMatrix = Matrix<Scalar, Dynamic, Dynamic, Order>;
  constexpr Index count = 24;
  Scalar lhs_storage[count], rhs_storage[count];
  for (Index i = 0; i < count; ++i) {
    lhs_storage[i] = Scalar(i + 1);
    rhs_storage[i] = Scalar(2 * i + 1);
  }
  DstMatrix expected(count, count), result(count, count);
  for (Index j = 0; j < count; ++j)
    for (Index i = 0; i < count; ++i) expected(i, j) = lhs_storage[i] * rhs_storage[j];
  for (Index stride : blas_singleton_strides(count)) {
    Map<const ColMatrix, 0, OuterStride<>> lhs(lhs_storage, count, 1, OuterStride<>(stride));
    Map<const RowMatrix, 0, OuterStride<>> rhs(rhs_storage, 1, count, OuterStride<>(stride));
    result.noalias() = lhs * rhs;
    VERIFY_IS_CWISE_EQUAL(result, expected);
    result.setOnes();
    result.noalias() += Scalar(2) * lhs * rhs;
    VERIFY_IS_CWISE_EQUAL(result, (DstMatrix::Ones(count, count) + Scalar(2) * expected).eval());

    Scalar scalar_storage = Scalar(2);
    Map<const RowMatrix, 0, OuterStride<>> scalar(&scalar_storage, 1, 1, OuterStride<>(stride));
    result.noalias() = lhs * scalar;
    for (Index i = 0; i < count; ++i) VERIFY_IS_EQUAL(result(i, 0), Scalar(2) * lhs_storage[i]);
  }
}

#ifdef EIGEN_USE_BLAS
// Call the adapters directly: public products redirect singleton outputs through dot products or GEMV.
template <typename Scalar, int Order>
void blas_direct_singleton_gemv() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  constexpr Index count = 24;
  const Index rows = Order == ColMajor ? count : 1;
  const Index cols = Order == ColMajor ? 1 : count;
  MatrixType lhs(rows, cols);
  VectorType rhs(cols);
  for (Index i = 0; i < lhs.size(); ++i) lhs.data()[i] = blas_stride_value<Scalar>(i);
  for (Index i = 0; i < cols; ++i) rhs(i) = blas_stride_value<Scalar>(i % 3);
  for (Index stride : blas_singleton_strides(count)) {
    VectorType result = VectorType::Ones(rows), expected = result;
    for (Index i = 0; i < rows; ++i)
      for (Index j = 0; j < cols; ++j) expected(i) += Scalar(2) * lhs(i, j) * rhs(j);
    internal::general_matrix_vector_product_gemv<Index, Scalar, Order, false, Scalar, false>::run(
        rows, cols, lhs.data(), stride, rhs.data(), cols == 1 ? stride : Index(1), result.data(),
        rows == 1 ? stride : Index(1), Scalar(2));
    VERIFY_IS_CWISE_EQUAL(result, expected);
  }
}

template <typename Scalar, int LhsOrder, int RhsOrder>
void blas_direct_singleton_gemm() {
  using Lhs = Matrix<Scalar, Dynamic, Dynamic, LhsOrder>;
  using Rhs = Matrix<Scalar, Dynamic, Dynamic, RhsOrder>;
  using Dst = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  const Index shapes[][3] = {{24, 1, 24}, {1, 24, 24}, {24, 24, 1}, {1, 1, 24}};
  for (const auto& shape : shapes) {
    const Index rows = shape[0], cols = shape[1], depth = shape[2];
    Lhs lhs(rows, depth);
    Rhs rhs(depth, cols);
    for (Index i = 0; i < lhs.size(); ++i) lhs.data()[i] = blas_stride_value<Scalar>(i % 3);
    for (Index i = 0; i < rhs.size(); ++i) rhs.data()[i] = blas_stride_value<Scalar>(i % 5);
    Dst expected = Dst::Ones(rows, cols);
    for (Index j = 0; j < cols; ++j)
      for (Index i = 0; i < rows; ++i)
        for (Index k = 0; k < depth; ++k) expected(i, j) += Scalar(2) * lhs(i, k) * rhs(k, j);
    for (Index stride : blas_singleton_strides(Index(24))) {
      Dst result = Dst::Ones(rows, cols);
      const Index lhs_stride = lhs.outerSize() == 1 ? stride : lhs.outerStride();
      const Index rhs_stride = rhs.outerSize() == 1 ? stride : rhs.outerStride();
      internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic> blocking(rows, cols, depth, 1,
                                                                                                  false);
      internal::general_matrix_matrix_product<Index, Scalar, LhsOrder, false, Scalar, RhsOrder, false, ColMajor,
                                              1>::run(rows, cols, depth, lhs.data(), lhs_stride, rhs.data(), rhs_stride,
                                                      result.data(), Index(1),
                                                      cols == 1 ? stride : result.outerStride(), Scalar(2), blocking,
                                                      nullptr);
      VERIFY_IS_CWISE_EQUAL(result, expected);
    }
  }
}
template <typename Scalar, int Order>
void blas_direct_trimmed_trmv() {
  using Vector = Matrix<Scalar, Dynamic, 1>;
  constexpr Index count = 24;
  constexpr int mode = Order == ColMajor ? Lower : Upper;
  const Index rows = Order == ColMajor ? 1 : count;
  const Index cols = Order == ColMajor ? count : 1;
  // Only (0,0) belongs to the triangular part; the other coefficients are implicit zeros.
  const Scalar coefficient = blas_stride_value<Scalar>(0);
  const Vector rhs = Vector::Ones(cols);
  Vector expected = Vector::Ones(rows);
  expected(0) += Scalar(2) * coefficient;
  for (Index stride : blas_singleton_strides(count)) {
    Vector result = Vector::Ones(rows);
    internal::triangular_matrix_vector_product_trmv<Index, mode, Scalar, false, Scalar, false, Order>::run(
        rows, cols, &coefficient, stride, rhs.data(), Index(1), result.data(), Index(1), Scalar(2));
    VERIFY_IS_CWISE_EQUAL(result, expected);
  }
}
#endif

template <typename Scalar, int Order>
void blas_structured_singleton_stride() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using ColMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  constexpr Index count = 24;
  const ColMatrix diagonal = Scalar(2) * ColMatrix::Identity(count, count);
  MatrixType storage(Order == ColMajor ? count : 1, Order == ColMajor ? 1 : count);
  for (Index i = 0; i < count; ++i) storage.data()[i] = blas_stride_value<Scalar>(i);
  for (Index stride : blas_singleton_strides(count)) {
    Map<const MatrixType, 0, OuterStride<>> input(storage.data(), storage.rows(), storage.cols(),
                                                  OuterStride<>(stride));
    MatrixType output_storage = storage;
    Map<MatrixType, 0, OuterStride<>> output(output_storage.data(), storage.rows(), storage.cols(),
                                             OuterStride<>(stride));
    EIGEN_IF_CONSTEXPR (Order == ColMajor) {
      diagonal.template triangularView<Lower>().solveInPlace(output);
      VERIFY_IS_CWISE_EQUAL(output, (storage * Scalar(0.5)).eval());
      ColMatrix result = diagonal.template selfadjointView<Lower>() * input;
      VERIFY_IS_CWISE_EQUAL(result, (storage * Scalar(2)).eval());
      result = ColMatrix::Zero(count, count);
      result.template selfadjointView<Lower>().rankUpdate(input);
      for (Index j = 0; j < count; ++j)
        for (Index i = j; i < count; ++i) VERIFY_IS_EQUAL(result(i, j), storage(i, 0) * numext::conj(storage(j, 0)));
    } else {
      diagonal.template triangularView<Lower>().template solveInPlace<OnTheRight>(output);
      VERIFY_IS_CWISE_EQUAL(output, (storage * Scalar(0.5)).eval());
      ColMatrix result = input * diagonal.template selfadjointView<Lower>();
      VERIFY_IS_CWISE_EQUAL(result, (storage * Scalar(2)).eval());
    }
  }
}

template <typename Scalar, int Order>
void blas_stride_tests() {
  blas_singleton_stride<Scalar, Order>();
  blas_singleton_outer_product<Scalar, Order>();
  blas_structured_singleton_stride<Scalar, Order>();
#ifdef EIGEN_USE_BLAS
  blas_direct_singleton_gemv<Scalar, Order>();
  blas_direct_trimmed_trmv<Scalar, Order>();
  blas_direct_singleton_gemm<Scalar, Order, ColMajor>();
  blas_direct_singleton_gemm<Scalar, Order, RowMajor>();
#endif
}

EIGEN_DECLARE_TEST(blas_singleton_stride) {
  blas_stride_helpers();
  blas_stride_tests<float, ColMajor>();
  blas_stride_tests<float, RowMajor>();
  blas_stride_tests<double, ColMajor>();
  blas_stride_tests<double, RowMajor>();
  blas_stride_tests<std::complex<float>, ColMajor>();
  blas_stride_tests<std::complex<float>, RowMajor>();
  blas_stride_tests<std::complex<double>, ColMajor>();
  blas_stride_tests<std::complex<double>, RowMajor>();
}
