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
    Eigen::MatrixXcd Mxy = Eigen::MatrixXcd::Constant(v.rows(),v.cols(),0.0);

    // Generate magnetizations
    Mxy(Eigen::all,Eigen::all) = M0*((i*(PI/VENC)*v).array().exp());
    // for (int k=0; k<3; k++){
    //   Mxy(Eigen::all,k) = M0*((i*(PI/VENC)*v.col(k)).array().exp());
    // }

  return Mxy;
}


// Fourier exponential
Eigen::MatrixXcd FourierExp(const Eigen::MatrixXd &r,
  const double &kx,
  const double &ky,
  const double &kz){

    // Complex unit
    const std::complex<double> i(0, 1);

    // Real and imaginary magnetizations
    Eigen::MatrixXcd tmp = Eigen::MatrixXcd::Constant(r.rows(),1,0.0);
    Eigen::MatrixXcd Exp = Eigen::MatrixXcd::Constant(r.rows(),3,0.0);

    // Generate magnetizations
    // Obs: the 2*pi term is not in the exponential because is included
    // in (kx,ky,kz), which were multiplied in the function FlowImage
    tmp(Eigen::all,0) = (-i*(r.col(0)*kx + r.col(1)*ky + r.col(2)*kz)).array().exp();
    for (size_t j = 0; j < Exp.cols(); j++){
      Exp(Eigen::all,j) = tmp;
    }

  return Exp;
}


// Calculate image kspace
Eigen::MatrixXcd FlowImage3D(
  const Eigen::SparseMatrix<double> &M, // Mass matrix
  const std::vector<Eigen::MatrixXd> &k, // kspace trajectory
  const Eigen::MatrixXd &t,       // kspace timings
  const Eigen::MatrixXd &v,       // object velocity
  const Eigen::MatrixXd &r0,      // object position
  const double &T2,               // T2 time of the blood
  const double &VENC,             // Velocity encoding
  const double &kz){             

    // Number of kspace lines/spokes/interleaves
    const size_t nb_lines = k[0].cols();
    
    // Number of measurements in the readout direction
    const size_t nb_meas = k[0].rows();

    // Number of spins
    const size_t nb_spins = r0.rows();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Eigen::MatrixXd kx = k[0]*(2.0*PI);
    const Eigen::MatrixXd ky = k[1]*(2.0*PI);
    const double kzz = kz*(2.0*PI);

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    Eigen::MatrixXd r = r0;

    // Build magnetizations and Fourier exponential
    Eigen::MatrixXcd Mag = FlowMags(v, VENC);
    Eigen::MatrixXcd fe = Eigen::MatrixXcd::Zero(nb_spins, 1);

    // Dummy ones for temporal integration
    Eigen::MatrixXd ones = Eigen::MatrixXd::Constant(1,nb_spins,1.0);
  
    // Image kspace
    Eigen::MatrixXcd Mxy = Eigen::MatrixXcd::Zero(nb_meas,nb_lines);

    // Iterate over kspace measurements/kspace points
    for (size_t j = 0; j < nb_lines; j++){
      for (size_t i = 0; i < nb_meas; i++){

        // Update blood position at time t(i,j)
        r.noalias() = r0 + v*t(i,j);

        // Fourier exponential
        fe.noalias() = FourierExp(r, kx(i,j), ky(i,j), kzz);

        // Integral calculation
        Mxy(i,j) = (ones*(M*Mag.col(2).cwiseProduct(fe)))[0];

        // Adds the T2 decay
        Mxy(i,j) *= std::exp(-t(i,j)/T2);

      }
    }

    return Mxy;
}


// Calculate image kspace
std::vector<Eigen::MatrixXcd> FlowImage3Dv2(
  const Eigen::SparseMatrix<double> &M, // Mass matrix
  const std::vector<Eigen::MatrixXd> &k, // kspace trajectory
  const Eigen::VectorXd &kz,
  const Eigen::MatrixXd &t,       // kspace timings
  const double &vel_enc_time,     // duration time of the bipolar gradients
  const Eigen::MatrixXd &v,       // object velocity
  const Eigen::MatrixXd &r0,      // object position
  const double &T2,               // T2 time of the blood
  const double &VENC){            // Velocity encoding             

    // Number of kspace lines/spokes/interleaves
    const size_t nb_lines = k[0].cols();
    
    // Number of measurements in the readout direction
    const size_t nb_meas = k[0].rows();

   // Number of measurements in the kz direction
    const size_t nb_kz = kz.size();

    // Number of spins
    const size_t nb_spins = r0.rows();

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

    // Dummy ones for temporal integration
    Eigen::MatrixXd ones = Eigen::MatrixXd::Constant(1,nb_spins,1.0);
  
    // Image kspace
    std::vector<Eigen::MatrixXcd> Mxy(nb_kz);
    for (size_t k = 0; k < nb_kz; k++){
      Mxy[k] = Eigen::MatrixXcd::Zero(nb_meas,nb_lines);
    }

    // Iterate over kspace measurements/kspace points
    for (size_t k = 0; k < nb_kz; k++){

      // Debugging
      py::print("    kz location ", k);

      for (size_t j = 0; j < nb_lines; j++){
        for (size_t i = 0; i < nb_meas; i++){

          // Update blood position at time t(i,j)
          r.noalias() = r0 + v*(t(i,j) - vel_enc_time);

          // Fourier exponential
          fe.noalias() = FourierExp(r, kx(i,j), ky(i,j), kzz(k));

          // Integral calculation
          Mxy[k](i,j) = (ones*(M*Mag.col(2).cwiseProduct(fe)))[0];

          // Adds the T2 decay
          Mxy[k](i,j) *= std::exp(-t(i,j)/T2);

        }
      }
    }

    return Mxy;
}


// Calculate image kspace
ComplexTensor FlowImage3Dv3(
  const Eigen::SparseMatrix<double> &M, // Mass matrix
  const std::vector<Eigen::MatrixXd> &k, // kspace trajectory
  const Eigen::VectorXd &kz,
  const Eigen::MatrixXd &t,       // kspace timings
  const Eigen::MatrixXd &v,       // object velocity
  const Eigen::MatrixXd &r0,       // object position
  const double &T2,               // T2 time of the blood
  const double &VENC){             // Velocity encoding             

    // Number of kspace lines/spokes/interleaves
    const size_t nb_lines = k[0].cols();
    
    // Number of measurements in the readout direction
    const size_t nb_meas = k[0].rows();

   // Number of measurements in the kz direction
    const size_t nb_kz = kz.size();

    // Number of spins
    const size_t nb_spins = r0.rows();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Eigen::MatrixXd kx = k[0]*(2.0*PI);
    const Eigen::MatrixXd ky = k[1]*(2.0*PI);
    const Eigen::VectorXd kzz = kz*(2.0*PI);

    // Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
    Eigen::MatrixXd r = r0;

    // Build magnetizations and Fourier exponential
    Eigen::MatrixXcd Mag = FlowMags(v, VENC);
    Eigen::MatrixXcd fe = Eigen::MatrixXcd::Zero(nb_spins, Mag.cols());

    // Dummy ones for temporal integration
    Eigen::MatrixXd ones = Eigen::MatrixXd::Constant(1,nb_spins,1.0);

    // Vector to store the k-space value for each velocity encoding
    Eigen::MatrixXcd k_vec = Eigen::MatrixXcd::Constant(1, 3, 0.0);

    // Image kspace
    ComplexTensor Mxy(nb_meas, nb_lines, nb_kz, 3);

    // Iterate over kspace measurements/kspace points
    for (size_t k = 0; k < nb_kz; k++){

      // Debugging
      py::print("    kz location ", k);

      for (size_t j = 0; j < nb_lines; j++){
        for (size_t i = 0; i < nb_meas; i++){

          // Update blood position at time t(i,j)
          r.noalias() = r0 + v*t(i,j);

          // Fourier exponential
          fe.noalias() = FourierExp(r, kx(i,j), ky(i,j), kzz(k));

          // Calculate k-space values, add T2* decay, and assign value to output array
          k_vec.noalias() = ones*(M*Mag.cwiseProduct(fe));
          k_vec *= std::exp(-t(i,j)/T2);
          for (size_t l = 0; l < 3; l++){
            Mxy(i,j,k,l) = k_vec(l); 
          }
        }
      }
    }

    return Mxy;
}


PYBIND11_MODULE(FlowToImage, m) {
    m.doc() = "Utilities for 4d flow image generation";
    m.def("FlowImage3D", &FlowImage3D);
    m.def("FlowImage3Dv2", &FlowImage3Dv2);
    m.def("FlowImage3Dv3", &FlowImage3Dv3);
}
