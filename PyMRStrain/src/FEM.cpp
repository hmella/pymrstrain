#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Custom types
typedef std::tuple<Eigen::Matrix<double, 4, 1>,
  Eigen::Matrix<double, 4, 1>,
  Eigen::Matrix<double, 4, 1>,
  Eigen::Matrix<double, 4, 1>> shapeFunctions;
typedef std::tuple<Eigen::Matrix<double, 4, 3>, std::array<double, 4>> quadRule;
typedef std::tuple<Eigen::Matrix<double, 4, 1>,
  Eigen::Matrix<double, 4, 1>,
  Eigen::Matrix<double, 4, 1>,
  Eigen::Matrix<double, 4, 1>,
  double> isopMap;


// Shape functions and derivatives
shapeFunctions P1ShapeTetra(double r, double s, double t){
  // Shape functions
  Eigen::Matrix<double, 4, 1> S;
  S(0,0) = 1-r-s-t;
  S(1,0) = r;
  S(2,0) = s;
  S(3,0) = t;

  // Derivatives
  Eigen::Matrix<double, 4, 1> dSdr;
  dSdr(0,0) = -1;
  dSdr(1,0) = 1;
  dSdr(2,0) = 0;
  dSdr(3,0) = 0;
  Eigen::Matrix<double, 4, 1> dSds;
  dSds(0,0) = -1;
  dSds(1,0) = 0;
  dSds(2,0) = 1;
  dSds(3,0) = 0;
  Eigen::Matrix<double, 4, 1> dSdt;
  dSdt(0,0) = -1;
  dSdt(1,0) = 0;
  dSdt(2,0) = 0;
  dSdt(3,0) = 1;

  return std::make_tuple(S, dSdr, dSds, dSdt);
}

// Isoparametric mapping
isopMap isopMapping(const Eigen::MatrixXd &x,
  const Eigen::MatrixXd &y,
  const Eigen::MatrixXd &z,
  const double &r,
  const double &s,
  const double &t){
  // Evaluate shape functions
  shapeFunctions sss = P1ShapeTetra(r,s,t);
  const Eigen::Matrix<double, 4, 1> S = std::get<0>(sss);
  const Eigen::Matrix<double, 4, 1> dSdr = std::get<1>(sss);
  const Eigen::Matrix<double, 4, 1> dSds = std::get<2>(sss);
  const Eigen::Matrix<double, 4, 1> dSdt = std::get<3>(sss);

  // Jacobian matrix
  Eigen::Matrix<double, 4, 4> J;
  J(0,0) = (dSdr.transpose()*x).sum();
  J(0,1) = (dSdr.transpose()*y).sum();
  J(0,2) = (dSdr.transpose()*z).sum();
  J(1,0) = (dSds.transpose()*x).sum();
  J(1,1) = (dSds.transpose()*y).sum();
  J(1,2) = (dSds.transpose()*z).sum();
  J(2,0) = (dSdt.transpose()*x).sum();
  J(2,1) = (dSdt.transpose()*y).sum();
  J(2,2) = (dSdt.transpose()*z).sum();

  // Jacobian determinant
  const double detJ = J.determinant();

  // Jacobian inverse matrix
  const Eigen::Matrix<double, 4, 4> invJ = J.inverse();

  // Shape functions in the global element
  const Eigen::Matrix<double, 4, 1> dSdx = invJ(0,0)*dSdr + invJ(0,1)*dSds + invJ(0,2)*dSdt;
  const Eigen::Matrix<double, 4, 1> dSdy = invJ(1,0)*dSdr + invJ(1,1)*dSds + invJ(1,2)*dSdt;
  const Eigen::Matrix<double, 4, 1> dSdz = invJ(2,0)*dSdr + invJ(2,1)*dSds + invJ(2,2)*dSdt;

  return std::make_tuple(S, dSdx, dSdy, dSdz, detJ);
}

// Quadrature rule
quadRule quadratureRule(int degree){
  // Points
  Eigen::Matrix<double, 4, 3> x;
  double a = 0.585410196624969;
  double b = 0.138196601125011;
  x(0,0) = a;
  x(0,1) = b;
  x(0,2) = b;
  x(1,0) = b;
  x(1,1) = a;
  x(1,2) = b;
  x(2,0) = b;
  x(2,1) = b;
  x(2,2) = a;
  x(3,0) = b;
  x(3,1) = b;
  x(3,2) = b;

  // Weights
  std::array<double, 4> w;
  w[0] = 1.0/24.0;
  w[1] = 1.0/24.0;
  w[2] = 1.0/24.0;
  w[3] = 1.0/24.0;

  return std::make_tuple(x, w);
}

// Assemble integral locally
std::array<double, 3> local_assemble(const Eigen::Matrix<double, 4, 3> &xe,
  const Eigen::Matrix<double, 4, 3> &values,
  const Eigen::Matrix<double, 4, 3> &qpoints,
  const std::array<double, 4> &qweights){

  // Assembled integral
  std::array<double, 3> integral;

  // Assembling
  for (int q=0; q<4; q++){ 
    // Isoparametric mapping
    const isopMap iso = isopMapping(xe.col(0),xe.col(1),xe.col(2),
      qpoints(q,0),qpoints(q,1),qpoints(q,2));
    const auto S = std::get<0>(iso);
    const auto detJ = std::get<4>(iso);

    // Matrix assembling (should the determinant be adjusted?)
    integral[0] = (S.array()*values.col(0).array()).sum()*detJ*qweights[q];
    integral[1] = (S.array()*values.col(1).array()).sum()*detJ*qweights[q];
    integral[2] = (S.array()*values.col(2).array()).sum()*detJ*qweights[q];
  }

  return integral;
}

// Assemble integral over the whole domain
std::array<double, 3> assemble(const Eigen::MatrixXd &elems,
  const Eigen::MatrixXd &nodes,
  const Eigen::MatrixXd &values){

  // Number of elements
  const int nel = elems.rows();

  // Quadrature rule
  const auto quadrature = quadratureRule(0);
  const auto qpoints  = std::get<0>(quadrature);
  const auto qweights = std::get<1>(quadrature);

  // Assembled integral
  std::array<double, 3> integral;

  // Assembling
  for (int e=0; e<nel; e++){ // loop over elements
    for (int n=0; n<4; n++){ // loop over nodes in the element e
      integral[0] += 1;
      integral[1] += 1;
      integral[2] += 1;      
    }
  }

  return integral;
}


PYBIND11_MODULE(FEM, m) {
    m.doc() = "Finite elements functions"; // optional module docstring
    m.def("assemble", &assemble, py::return_value_policy::reference);    
}
