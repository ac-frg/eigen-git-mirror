// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR double
#endif

template <template <typename, int> class Decomposition, int Size, int Order, int UpLo>
static void BM_CholeskyCompute(benchmark::State& state) {
  using Scalar = SCALAR;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Size, Size, Order>;
  const Index size = Size == Dynamic ? Index(state.range(0)) : Index(Size);
  MatrixType basis = MatrixType::Random(size, size);
  MatrixType input = basis * basis.adjoint();
  input.diagonal().array() += RealScalar(size);
  Decomposition<MatrixType, UpLo> factor(size);
  factor.compute(input);
  const RealScalar bound = RealScalar(32 * size) * NumTraits<RealScalar>::epsilon() * input.norm();
  if (factor.info() != Success || !((factor.reconstructedMatrix() - input).norm() <= bound)) {
    state.SkipWithError("Cholesky reconstruction failed");
    return;
  }

  benchmark::DoNotOptimize(input.data());
  for (auto _ : state) {
    factor.compute(input);
    benchmark::DoNotOptimize(factor);
    benchmark::ClobberMemory();
  }
}

#define EIGEN_BENCH_SMALL_CHOLESKY(Decomposition, Order, UpLo)                \
  BENCHMARK_TEMPLATE(BM_CholeskyCompute, Decomposition, 2, Order, UpLo);      \
  BENCHMARK_TEMPLATE(BM_CholeskyCompute, Decomposition, 3, Order, UpLo);      \
  BENCHMARK_TEMPLATE(BM_CholeskyCompute, Decomposition, 4, Order, UpLo);      \
  BENCHMARK_TEMPLATE(BM_CholeskyCompute, Decomposition, 6, Order, UpLo);      \
  BENCHMARK_TEMPLATE(BM_CholeskyCompute, Decomposition, 8, Order, UpLo);      \
  BENCHMARK_TEMPLATE(BM_CholeskyCompute, Decomposition, Dynamic, Order, UpLo) \
      ->DenseRange(1, 8)                                                      \
      ->Arg(16)                                                               \
      ->Arg(32)                                                               \
      ->Arg(48)                                                               \
      ->Arg(64)

EIGEN_BENCH_SMALL_CHOLESKY(LLT, ColMajor, Lower);
EIGEN_BENCH_SMALL_CHOLESKY(LLT, ColMajor, Upper);
EIGEN_BENCH_SMALL_CHOLESKY(LLT, RowMajor, Lower);
EIGEN_BENCH_SMALL_CHOLESKY(LLT, RowMajor, Upper);
EIGEN_BENCH_SMALL_CHOLESKY(LDLT, ColMajor, Lower);
EIGEN_BENCH_SMALL_CHOLESKY(LDLT, ColMajor, Upper);
EIGEN_BENCH_SMALL_CHOLESKY(LDLT, RowMajor, Lower);
EIGEN_BENCH_SMALL_CHOLESKY(LDLT, RowMajor, Upper);

#undef EIGEN_BENCH_SMALL_CHOLESKY
