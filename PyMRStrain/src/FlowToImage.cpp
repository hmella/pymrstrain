#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>


using namespace Eigen;
namespace py = pybind11;

// Definition of PI and complex unit
const float PI = M_PI;
const std::complex<float> i1(0, 1);

// Definition of magnetization datatype for 4d flow expression
typedef std::tuple<MatrixXcf, MatrixXcf> magnetizations;

// Definition of third rank complex tensor datatype
typedef Tensor<std::complex<float>, 4> ComplexTensor;

// Calculate image kspace
ComplexTensor FlowImage3D(
  const int &MPI_rank,
  const SparseMatrix<float> &M, // Mass matrix
  const std::vector<MatrixXf> &kxy, // kspace trajectory
  const VectorXf &kzz,
  const MatrixXf &t,  // kspace timings
  const MatrixXf &v,  // object velocity
  const MatrixXf &r0, // object position
  const float &T2,          // T2 time of the blood
  const float &VENC){       // Velocity encoding

    // Number of kspace lines/spokes/interleaves
    const uint nb_lines = kxy[0].cols();
    
    // Number of measurements in the readout direction
    const uint nb_meas = kxy[0].rows();

   // Number of measurements in the kz direction
    const uint nb_kz = kzz.size();

    // Number of spins
    const uint nb_spins = r0.rows();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const MatrixXf kx = 2.0 * PI * kxy[0];
    const MatrixXf ky = 2.0 * PI * kxy[1];
    const VectorXf kz = 2.0 * PI * kzz;

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    MatrixXf r(nb_spins, 3);

    // Kspace and Fourier exponential
    const MatrixXcf Mxy = 1.0e+3 * nb_spins * (i1 * PI / VENC * v).array().exp();
    VectorXf fe_xy(nb_spins);
    VectorXcf fe(nb_spins);

    // kspace
    ComplexTensor kspace(nb_meas, nb_lines, nb_kz, 3);

    // T2* decay
    const MatrixXf T2_decay = (-t / T2).array().exp();

    // Iterate over kspace measurements/kspace points
    for (uint j = 0; j < nb_lines; ++j){

      // Debugging
      if (MPI_rank == 0){ py::print("  ky location ", j); }

      // Iterate over slice kspace measurements
      for (uint i = 0; i < nb_meas; ++i){

        // Update blood position at time t(i,j)
        r.noalias() = r0 + v*t(i,j);

        // Fourier exponential
        fe_xy.noalias() = -(r.col(0) * kx(i,j) + r.col(1) * ky(i,j));

        for (uint k = 0; k < nb_kz; ++k){

          // Update Fourier exponential
          fe(indexing::all, 0) = (i1 * (fe_xy - r.col(2) * kz(k))).array().exp();

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (uint l = 0; l < 3; ++l){
            kspace(i,j,k,l) = (M * Mxy.col(l).cwiseProduct(fe)).sum() * T2_decay(i,j);
          }
        }
      }
    }

    return kspace;
}


PYBIND11_MODULE(FlowToImage, m) {
    m.doc() = "Utilities for 4d flow image generation";
    m.def("FlowImage3D", &FlowImage3D);
}
