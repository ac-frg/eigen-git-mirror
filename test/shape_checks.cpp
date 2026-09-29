// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Dense>
#include <contrib/Eigen/MatrixFunctions>

struct MissingShape {};
struct PartialShape {};
template <int Rows, int Cols>
struct ShapeProbe;
namespace Eigen {
namespace internal {
template <int Rows, int Cols>
struct traits<ShapeProbe<Rows, Cols>> : traits<Matrix<double, Rows, Cols>> {};
template <>
struct traits<PartialShape> {
  static constexpr int RowsAtCompileTime = 3;
};
}  // namespace internal
}  // namespace Eigen

template <int Rows, int Cols>
struct ShapeProbe : MatrixBase<ShapeProbe<Rows, Cols>> {
  using Nested = const ShapeProbe&;
  mutable int row_reads = 0;
  mutable int col_reads = 0;
  Index rows() const {
    ++row_reads;
    return 2;
  }
  Index cols() const {
    ++col_reads;
    return 2;
  }
};

void rank_update_shape_checks() {
  Matrix2d b = Matrix2d::Constant(2);
  Matrix2d updated = Matrix2d::Zero();
  updated.selfadjointView<Lower>().rankUpdate(b);
  VERIFY_IS_EQUAL(Matrix2d(updated.selfadjointView<Lower>()), (b * b.transpose()).eval());
  MatrixXd wrong = MatrixXd::Ones(3, 2);
  VERIFY_RAISES_ASSERT(updated.selfadjointView<Lower>().rankUpdate(wrong));
#ifdef EIGEN_NO_STATIC_ASSERT
  Matrix<double, 3, 2> fixed_wrong = wrong;
  VERIFY_RAISES_ASSERT(updated.selfadjointView<Lower>().rankUpdate(fixed_wrong));
#endif
}

void product_shape_checks() {
  rank_update_shape_checks();
  ShapeProbe<2, 2> a, b;
  const auto product = a * b;
  EIGEN_UNUSED_VARIABLE(product);
  VERIFY_IS_EQUAL(a.col_reads + b.row_reads, 0);
  const MatrixBase<ShapeProbe<2, 2>>& base_a = a;
  const MatrixBase<ShapeProbe<2, 2>>& base_b = b;
  EIGEN_ASSERT_PRODUCT_SHAPE(base_a, base_b);
  VERIFY_IS_EQUAL(a.col_reads + b.row_reads, 0);
  Matrix2d fixed = Matrix2d::Identity();
  MatrixXd wrong(3, 2);
  VERIFY_RAISES_ASSERT((fixed * wrong));
#ifdef EIGEN_NO_STATIC_ASSERT
  Matrix<double, 3, 2> fixed_wrong;
  VERIFY_RAISES_ASSERT((Product<Matrix2d, decltype(fixed_wrong)>(fixed, fixed_wrong)));
#endif
  Transpositions<2> swaps;
  swaps.setIdentity();
  VERIFY_IS_EQUAL((swaps * fixed).eval(), fixed);
  VERIFY_IS_EQUAL((fixed * swaps.transpose()).eval(), fixed);
}

template <int Size, int Order>
void product_destination_shape_checks() {
  using MatrixType = Matrix<double, Size, Size, Order>;
  const MatrixType identity = MatrixType::Identity(3, 3);
  const MatrixType values = MatrixType::Constant(3, 3, 2);
  MatrixType result = MatrixType::Zero(3, 3);
  result.noalias() += identity * values;
  VERIFY_IS_EQUAL(result, values);
  result.noalias() -= identity * values;
  VERIFY(result.isZero());
  result.noalias() += identity.template selfadjointView<Lower>() * values;
  VERIFY_IS_EQUAL(result, values);
  result.noalias() -= identity.template triangularView<Upper>() * values;
  VERIFY(result.isZero());

  PermutationMatrix<Size> permutation(3);
  permutation.setIdentity();
  DiagonalMatrix<double, Size> diagonal(3);
  diagonal.diagonal().setConstant(2);
  result.noalias() = identity.template triangularView<Lower>() * diagonal;
  VERIFY_IS_EQUAL(result, 2 * identity);
  result.noalias() = diagonal * identity.template triangularView<Upper>();
  VERIFY_IS_EQUAL(result, 2 * identity);
  result.noalias() = identity.template selfadjointView<Lower>() * diagonal;
  VERIFY_IS_EQUAL(result, 2 * identity);
  result.setZero();
  const auto scaled = permutation * diagonal;
  result += scaled;
  VERIFY_IS_EQUAL(result, 2 * identity);
  result -= scaled;
  VERIFY(result.isZero());
  MatrixXd wrong_destination = MatrixXd::Zero(2, 3);
  VERIFY_RAISES_ASSERT(wrong_destination.noalias() += identity * values);
  VERIFY_RAISES_ASSERT(wrong_destination.noalias() -= identity * values);
  VERIFY_RAISES_ASSERT(wrong_destination += scaled);
  VERIFY_RAISES_ASSERT(wrong_destination -= scaled);
  result.noalias() += scaled * identity;
  VERIFY_IS_EQUAL(result, 2 * identity);
  result.noalias() -= scaled * identity;
  VERIFY(result.isZero());
  VERIFY_IS_EQUAL((scaled * permutation).toDenseMatrix(), 2 * identity);
  VERIFY_IS_EQUAL((permutation * scaled).toDenseMatrix(), 2 * identity);
  DiagonalMatrix<double, Dynamic> wrong(2);
  VERIFY_RAISES_ASSERT(permutation * wrong);
}

EIGEN_DECLARE_TEST(shape_checks) {
  ShapeProbe<2, 2> fixed_a, fixed_b;
  const auto sum = fixed_a + fixed_b;
  EIGEN_UNUSED_VARIABLE(sum);
  internal::resize_if_allowed(fixed_a, fixed_b, internal::add_assign_op<double, double>());
  VERIFY_IS_EQUAL(fixed_a.row_reads + fixed_b.row_reads + fixed_a.col_reads + fixed_b.col_reads, 0);
  ShapeProbe<2, Dynamic> partial_a, partial_b;
  const auto partial_sum = partial_a + partial_b;
  EIGEN_UNUSED_VARIABLE(partial_sum);
  VERIFY_IS_EQUAL(partial_a.row_reads + partial_b.row_reads, 0);
#ifndef EIGEN_NO_DEBUG
  VERIFY(partial_a.col_reads > 0 && partial_b.col_reads > 0);
#endif
  Matrix2d matrix = Matrix2d::Identity();
  MatrixXd bad(3, 2);
  VERIFY_RAISES_ASSERT((matrix + bad));
  VERIFY_RAISES_ASSERT(matrix += bad);
  VERIFY_IS_EQUAL(hcat(matrix, matrix).cols(), 4);
  VERIFY_IS_EQUAL(vcat(matrix, matrix).rows(), 4);
  Matrix<double, 2, 3> concat_wide = Matrix<double, 2, 3>::Ones();
  VERIFY_IS_EQUAL(hcat(matrix, concat_wide).cols(), 5);
  VERIFY_IS_EQUAL(vcat(matrix, concat_wide.transpose()).rows(), 5);
#ifdef EIGEN_NO_STATIC_ASSERT
  VERIFY_RAISES_ASSERT(hcat(matrix, concat_wide.transpose()));
  VERIFY_RAISES_ASSERT(vcat(matrix, concat_wide));
#endif
  Matrix<double, 2, 3> wide = Matrix<double, 2, 3>::Ones();
  const DiagonalMatrix<double, 2> left_diagonal(Vector2d::Constant(2));
  const DiagonalMatrix<double, 3> right_diagonal(Vector3d::Constant(3));
  Matrix<double, 2, 3> triangular_product = left_diagonal * wide.triangularView<Upper>();
  VERIFY_IS_EQUAL(triangular_product, 2 * wide.triangularView<Upper>().toDenseMatrix());
  triangular_product = wide.triangularView<Upper>() * right_diagonal;
  VERIFY_IS_EQUAL(triangular_product, 3 * wide.triangularView<Upper>().toDenseMatrix());
  CALL_SUBTEST(product_shape_checks());
  CALL_SUBTEST((product_destination_shape_checks<3, ColMajor>()));
  CALL_SUBTEST((product_destination_shape_checks<3, RowMajor>()));
  CALL_SUBTEST((product_destination_shape_checks<Dynamic, ColMajor>()));
  CALL_SUBTEST((product_destination_shape_checks<Dynamic, RowMajor>()));
  using namespace Eigen::internal;
  STATIC_CHECK((dimension_relation<3, 3>::Proven));
  STATIC_CHECK((!dimension_relation<3, 2>::Compatible));
  STATIC_CHECK((dimension_relation<Dynamic, 3>::Compatible));
  STATIC_CHECK((!dimension_relation<Dynamic, 3>::Known));
  STATIC_CHECK((compile_time_shape<MissingShape>::Rows == Dynamic));
  STATIC_CHECK((compile_time_shape<PartialShape>::Rows == 3));
  STATIC_CHECK((compile_time_shape<PartialShape>::Cols == Dynamic));
  STATIC_CHECK((compile_time_shape<Matrix<double, Dynamic, 1, 0, 3, 1>>::Size == Dynamic));
  STATIC_CHECK((matching_sizes<Matrix<double, 2, 3>, Matrix<double, 3, 2>>::Proven));
  STATIC_CHECK((matching_shapes<Matrix<double, 1, 0>, Matrix<double, 0, 1>>::Compatible));
  const Matrix<double, 1, 0> empty_row;
  const Matrix<double, 0, 1> empty_column;
  VERIFY_RAISES_ASSERT(empty_row + empty_column);
  VERIFY_RAISES_ASSERT(internal::resize_if_allowed(empty_row, empty_column, internal::add_assign_op<double, double>()));
  STATIC_CHECK((square_dimensions<const Matrix3d&>::Proven));
  int reads = 0;
  struct RuntimeDimensions {
    mutable int accesses = 0;
    Index rows() const {
      ++accesses;
      return 2;
    }
    Index cols() const {
      ++accesses;
      return 3;
    }
  } runtime_a, runtime_b;
  EIGEN_ASSERT_SAME_ROWS(runtime_a, runtime_b);
  EIGEN_ASSERT_SAME_COLS(runtime_a, runtime_b);
#ifndef EIGEN_NO_DEBUG
  VERIFY_IS_EQUAL(runtime_a.accesses, 2);
  VERIFY_IS_EQUAL(runtime_b.accesses, 2);
#endif
  EIGEN_SHAPE_ASSERT((dimension_relation<3, 3>::Proven), ++reads == 1);
  VERIFY_IS_EQUAL(reads, 0);
#ifndef EIGEN_NO_DEBUG
  EIGEN_SHAPE_ASSERT((dimension_relation<Dynamic, 3>::Proven), ++reads == 1);
  VERIFY_IS_EQUAL(reads, 1);
  VERIFY_RAISES_ASSERT([] { EIGEN_SHAPE_ASSERT(false, false); }());
#endif
}
