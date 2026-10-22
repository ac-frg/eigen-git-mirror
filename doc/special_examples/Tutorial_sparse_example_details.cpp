// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/Sparse>
#include <vector>
#include <fstream>

typedef Eigen::SparseMatrix<double> SpMat;  // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs, Eigen::VectorXd& b,
                       const Eigen::VectorXd& boundary) {
  int n = int(boundary.size());
  int id1 = i + j * n;

  if (i == -1 || i == n)
    b(id) -= w * boundary(j);  // constrained coefficient
  else if (j == -1 || j == n)
    b(id) -= w * boundary(i);  // constrained coefficient
  else
    coeffs.push_back(T(id, id1, w));  // unknown coefficient
}

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n) {
  b.setZero();
  Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0, EIGEN_PI).sin().pow(2);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      int id = i + j * n;
      insertCoefficient(id, i - 1, j, -1, coefficients, b, boundary);
      insertCoefficient(id, i + 1, j, -1, coefficients, b, boundary);
      insertCoefficient(id, i, j - 1, -1, coefficients, b, boundary);
      insertCoefficient(id, i, j + 1, -1, coefficients, b, boundary);
      insertCoefficient(id, i, j, 4, coefficients, b, boundary);
    }
  }
}

bool saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename) {
  std::ofstream image(filename, std::ios::binary);
  if (!image) return false;
  // Binary PGM: https://netpbm.sourceforge.net/doc/pgm.html
  image << "P5\n" << n << ' ' << n << "\n255\n";
  const Eigen::Array<unsigned char, Eigen::Dynamic, 1> bits = (x * 255).cast<unsigned char>();
  image.write(reinterpret_cast<const char*>(bits.data()), static_cast<std::streamsize>(bits.size()));
  image.close();
  return bool(image);
}
