# include <iostream>

# include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Dynamic;
using Eigen::VectorXd;


/////////////////////////////////
// main
/////////////////////////////////

int main() {

  MatrixXd foo(2, 2);
  VectorXd v(2);
  VectorXd q(2);
  float z;

  foo << 1.0, 0.1, 0.101, 1.0;
  v << 5, 6;
  q << 7, 8;

  std::cout << v << std::endl;
  printf(".................\n");
  std::cout << q << std::endl;
  printf(".................\n");
  std::cout << v * q << std::endl; // Componentwise as expected
  printf(".................\n");

  std::cout << foo << std::endl;
  printf(".................\n");

  std::cout << v << std::endl;
  printf(".................\n");

  std::cout << foo * v << std::endl;
  printf(".................\n");

  z = v.dot(foo * v);
  std::cout << v.adjoint() * foo * v << std::endl;
  std::cout << v.dot(foo * v) << std::endl;
  std::cout << z << std::endl;
  printf(".................\n");

  std::cout << foo.row(0) << std::endl;
  std::cout << foo.block(0, 0, 1, 2) << std::endl;
  printf(".................\n");

  std::cout << foo.col(0) << std::endl;
  printf(".................\n");

  std::cout << v << std::endl;
  printf(".................\n");

  std::cout << v.dot(foo.col(0)) << std::endl;
  printf(".................\n");

  std::cout << foo.col(0) * v << std::endl; // Note: this does something strange.
  printf(".................\n");

  z = v.dot(foo * v);
  std::cout << foo.row(0) * v << std::endl;
  std::cout << v.dot(foo.col(0)) << std::endl;

  return 0;
}
