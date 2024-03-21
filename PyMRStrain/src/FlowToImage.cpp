#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;
namespace py = pybind11;

// Definition of PI and complex unit
const float PI = M_PI;
const std::complex<float> i1(0.0f, 1.0f);

// Definition of magnetization datatype for 4d flow expression
typedef std::tuple<MatrixXcf, MatrixXcf> magnetizations;

// Definition of fourth rank complex tensor datatype
typedef Tensor<std::complex<float>, 4> ComplexTensor;

// Definition of third rank complex tensor datatype
typedef Tensor<float, 3> klocations;

// Calculate image kspace
ComplexTensor FlowImage3D(
  const int &MPI_rank,
  const SparseMatrix<float> &M,        // Mass matrix
  const std::vector<klocations> &kloc, // kspace trajectory
  const MatrixXf &t,  // kspace timings
  const MatrixXf &v,  // object velocity
  const MatrixXf &r0, // object initial position
  const VectorXf &gamma_x_delta_B0,  // Field inhomogeneity
  const float &T2,          // T2 time of the blood
  const float &VENC,        // Velocity encoding
  const VectorXf &profile   // Slice profile
  ){

    // Number of measurements in the readout direction
    const uint nb_meas = kloc[0].dimension(0);

    // Number of kspace lines/spokes/interleaves
    const uint nb_lines = kloc[0].dimension(1);
    
   // Number of measurements in the kz direction
    const uint nb_kz = kloc[0].dimension(2);

    // Number of spins
    const uint nb_spins = r0.rows();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const klocations kx = 2.0 * PI * kloc[0];
    const klocations ky = 2.0 * PI * kloc[1];
    const klocations kz = 2.0 * PI * kloc[2];

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    MatrixXf r(nb_spins, 3);

    // Kspace, Fourier exponential, and off-resonance phase
    MatrixXcf Mxy = 1.0e+3 * nb_spins * (i1 * PI / VENC * v).array().exp();
    VectorXcf fe(nb_spins);
    VectorXf  phi_off(nb_spins);

    // Apply slice profile to magnetization
    for (uint s = 0; s < Mxy.cols(); ++s){
      Mxy(indexing::all,s) = Mxy.col(s).cwiseProduct(profile);
    }

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

        // Update off-resonance phase
        phi_off.noalias() = gamma_x_delta_B0*t(i,j);

        for (uint k = 0; k < nb_kz; ++k){

          // Update Fourier exponential
          fe(indexing::all,0) = (- i1 * (r.col(0)*kx(i,j,k) + r.col(1)*ky(i,j,k) + r.col(2)*kz(i,j,k) + phi_off)).array().exp();

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
