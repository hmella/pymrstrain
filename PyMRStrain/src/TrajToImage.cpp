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
Matrix3d ZRotation(const double theta){
  Matrix3d R;
  R(0,0) = std::cos(theta);
  R(0,1) = std::sin(theta);
  R(1,0) = -std::sin(theta);
  R(1,1) = std::cos(theta);
  R(2,2) = 1.0;
  return R;
}

// Calculate image kspace
MatrixXcd TrajToImage(const std::vector<MatrixXd> &traj,
                      const Ref<const MatrixXd> &time,
                      const Ref<const MatrixXd> &M0,
                      const Ref<const MatrixXd> &r,
                      const double T2){

    // Number of kspace lines/spokes/interleaves
    const size_t nb_lines = traj[0].cols();
    
    // Number of measurements in the readout direction
    const size_t nb_meas = traj[0].rows();

    // Number of spins
    const size_t nb_spins = r.rows();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    MatrixXd kx = traj[0]*(2.0*PI);
    MatrixXd ky = traj[1]*(2.0*PI);

    // Image kspace
    MatrixXcd Mxy = MatrixXcd::Zero(nb_meas,nb_lines);

    // Complex unit times 2*PI
    std::complex<double> cu(0, 1);

    // T2 decay
    double T2_decay = 0.0;

    // Rotation tensor
    // std::vector<Matrix3d> R;
    Matrix3d R;
    Vector3d tmp;

    // Iterate over kspace measurements/kspace points
    for (size_t j = 0; j < nb_lines; j++){
      for (size_t i = 0; i < nb_meas; i++){

        // Signal decay
        T2_decay = std::exp(-time(i,j)/T2);

        // Relaxation tensor1
        // R = ZRotation(r(all,0)*kx(i,j) + r(all,1)*ky(i,j));
        for (size_t k = 0; k < nb_spins; k++){
          R.noalias() = ZRotation(r(k,0)*kx(i,j) + r(k,1)*ky(i,j));
          tmp.noalias() = R.lazyProduct(M0(all,k));
          Mxy(i,j) += tmp(0) + cu*tmp(1);
        }

        // Adds the T2 decay
        Mxy(i,j) *= T2_decay;

      }
    }

    return Mxy;
}



PYBIND11_MODULE(TrajToImage, m) {
    m.doc() = "Utilities for spins-based image generation"; // optional module docstring
    m.def("TrajToImage", &TrajToImage);
}
