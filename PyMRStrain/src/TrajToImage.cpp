#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace Eigen;

// Definition of PI
const double PI = std::atan(1.0)*4.0;
const double GAMMA = 42.58;  // MHz/T

std::complex<double> fast_exp(double x){
  std::complex<double> out(std::cos(x), std::sin(x));
  return out;
}

// Calculate relaxation matrices
Matrix2d Rot(const double theta){
  Matrix2d R;
  const double s = std::sin(theta);
  const double c = std::cos(theta);
  R(0,0) = c;
  R(0,1) = s;
  R(1,0) = -s;
  R(1,1) = c;
  // R(2,2) = 1.0;
  return R;
}

// Calculate image kspace
MatrixXcd TrajToImage(const std::vector<MatrixXd> &k,
                      const MatrixXd &t,
                      const MatrixXd &M0,
                      const MatrixXd &r,
                      const double T2){

    // Number of kspace lines/spokes/interleaves
    const size_t nb_lines = k[0].cols();
    
    // Number of measurements in the readout direction
    const size_t nb_meas = k[0].rows();

    // Number of spins
    const size_t nb_spins = r.rows();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const MatrixXd kx = k[0]*(2.0*PI);
    const MatrixXd ky = k[1]*(2.0*PI);

    // Image kspace
    MatrixXcd Mxy = MatrixXcd::Zero(nb_meas,nb_lines);

    // Complex unit times 2*PI
    const std::complex<double> cu(0, 1);

    // Rotation tensor
    Matrix2d R;
    Vector2d tmp;

    // Iterate over kspace measurements/kspace points
    for (size_t j = 0; j < nb_lines; j++){
      for (size_t i = 0; i < nb_meas; i++){

        // Relaxation tensor1
        // R = Rot(r(all,0)*kx(i,j) + r(all,1)*ky(i,j));
        for (size_t s = 0; s < nb_spins; s++){
          R.noalias() = Rot(r(s,0)*kx(i,j) + r(s,1)*ky(i,j));
          tmp.noalias() = R*M0.col(s);
          Mxy(i,j) += tmp(0) + cu*tmp(1);
        }

        // Adds the T2 decay
        Mxy(i,j) *= std::exp(-t(i,j)/T2);

      }
    }

    return Mxy;
}



PYBIND11_MODULE(TrajToImage, m) {
    m.doc() = "Utilities for spins-based image generation"; // optional module docstring
    m.def("TrajToImage", &TrajToImage);
}
