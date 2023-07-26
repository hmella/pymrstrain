#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>


namespace py = pybind11;


// Definition of PI
const double PI = std::atan(1.0)*4.0;


// Definition of magnetization datatype for 4d flow expression
typedef std::tuple<Eigen::MatrixXcd,
  Eigen::MatrixXcd> magnetizations;

// Definition of third rank complex tensor datatype
typedef Eigen::Tensor<std::complex<double>, 4> ComplexTensor;


// 4D flow magnetizations (FourierExp can be added here)
Eigen::MatrixXcd FlowMags(const Eigen::MatrixXd &v,
  const double &VENC){

    // Complex unit
    const std::complex<double> i(0, 1);

    // Thermanl equilibrium magnetization
    const double M0 = 1000.0*v.rows();

    // Real and imaginary magnetizations
    Eigen::MatrixXcd Mxy = M0*((i*(PI/VENC)*v).array().exp());

  return Mxy;
}


// Fourier exponential
void FourierExp(Eigen::MatrixXcd& Exp,
  const Eigen::MatrixXd &r,
  const double &kx,
  const double &ky,
  const double &kz){

    // Complex unit
    const std::complex<double> i(0, 1);

    // Generate magnetizations
    // Obs: the 2*pi term is not in the exponential because is included
    // in (kx,ky,kz), which were multiplied in the function FlowImage
    Exp(Eigen::all,0) = (-i*(r.col(0)*kx + r.col(1)*ky + r.col(2)*kz)).array().exp();
}


// Calculate image kspace
ComplexTensor FlowImage3D(
  const int &MPI_rank,
  const Eigen::SparseMatrix<double> &M, // Mass matrix
  const std::vector<Eigen::MatrixXd> &k, // kspace trajectory
  const Eigen::VectorXd &kz,
  const Eigen::MatrixXd &t,       // kspace timings
  const Eigen::MatrixXd &v,       // object velocity
  const Eigen::MatrixXd &r0,       // object position
  const double &T2,               // T2 time of the blood
  const double &VENC){             // Velocity encoding             

    // Number of kspace lines/spokes/interleaves
    const uint nb_lines = k[0].cols();
    
    // Number of measurements in the readout direction
    const uint nb_meas = k[0].rows();

   // Number of measurements in the kz direction
    const uint nb_kz = kz.size();

    // Number of spins
    const uint nb_spins = r0.rows();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Eigen::MatrixXd kx = k[0]*(2.0*PI);
    const Eigen::MatrixXd ky = k[1]*(2.0*PI);
    const Eigen::VectorXd kzz = kz*(2.0*PI);

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    Eigen::MatrixXd r = r0;

    // Build magnetizations and Fourier exponential
    Eigen::MatrixXcd Mag = FlowMags(v, VENC);
    Eigen::MatrixXcd fe = Eigen::MatrixXcd::Zero(nb_spins, 1);

    // Image kspace
    ComplexTensor Mxy(nb_meas, nb_lines, nb_kz, 3);

    // T2* decay
    double T2_decay = 0.0;

    // Iterate over kspace measurements/kspace points
    for (uint j = 0; j < nb_lines; j++){

      // Debugging
      if (MPI_rank == 0){
        py::print("  ky location ", j);
      }

      // Iterate over slice kspace measurements
      for (uint i = 0; i < nb_meas; i++){

        // Update blood position at time t(i,j)
        r.noalias() = r0 + v*t(i,j);

        // T2* decay
        T2_decay = std::exp(-t(i,j)/T2);

        for (uint k = 0; k < nb_kz; k++){

          // Update Fourier exponential
          FourierExp(fe, r, kx(i,j), ky(i,j), kzz(k));

          // Calculate k-space values, add T2* decay, and assign value to output array
          for (uint l = 0; l < 3; l++){
            Mxy(i,j,k,l) = (M*Mag.col(l).cwiseProduct(fe)).sum()*T2_decay;
          }
        }
      }
    }

    return Mxy;
}


PYBIND11_MODULE(FlowToImage, m) {
    m.doc() = "Utilities for 4d flow image generation";
    m.def("FlowImage3D", &FlowImage3D);
}
