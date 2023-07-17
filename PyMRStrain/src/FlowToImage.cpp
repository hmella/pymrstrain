#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace py = pybind11;

// Definition of PI
const double PI = std::atan(1.0)*4.0;
const double GAMMA = 42.58;  // MHz/T

// Definition of 4dflow datatype
typedef std::tuple<Eigen::MatrixXcd,
  Eigen::MatrixXcd> magnetizations;


// 4D flow magnetizations (FourierExp can be added here)
Eigen::MatrixXcd FlowMags(const Eigen::MatrixXd &v,
  const double &VENC){

    // Complex unit
    const std::complex<double> i(0, 1);

    // Real and imaginary magnetizations
    Eigen::MatrixXcd Mxy = Eigen::MatrixXcd::Constant(v.rows(),v.cols(),0.0);

    // Generate magnetizations
    for (int k=0; k<3; k++){
      Mxy(Eigen::all,k) = ((i*(PI/VENC)*v.col(k)).array().exp());
    }

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
    Eigen::MatrixXcd Exp = Eigen::MatrixXcd::Constant(r.rows(),1,0.0);

    // Generate magnetizations
    Exp(Eigen::all,0) = (i*(r.col(0)*kx + r.col(1)*ky + r.col(2)*kz)).array().exp();

  return Exp;
}


// Calculate image kspace
Eigen::MatrixXcd FlowImage3D(
  Eigen::SparseMatrix<double> &M, // Mass matrix
  const std::vector<Eigen::MatrixXd> &k, // kspace trajectory
  const Eigen::MatrixXd &t,       // kspace timings
  const Eigen::MatrixXd &v,       // object velocity
  const Eigen::MatrixXd &r,       // object position
  const double &T2,               // T2 time of the blood
  const double &VENC,             // Velocity encoding
  const double &kz){             

    // Number of kspace lines/spokes/interleaves
    const size_t nb_lines = k[0].cols();
    
    // Number of measurements in the readout direction
    const size_t nb_meas = k[0].rows();

    // Number of spins
    const size_t nb_spins = r.rows();

    // Get the equivalent gradient needed to go from the center of the kspace
    // to each location
    const Eigen::MatrixXd kx = k[0]*(2.0*PI);
    const Eigen::MatrixXd ky = k[1]*(2.0*PI);
    const double kzz = kz*(2.0*PI);

    // Build magnetizations and F
    Eigen::MatrixXcd Mag = FlowMags(v, VENC);

    // Dummy ones for temporal integration
    Eigen::MatrixXd ones = Eigen::MatrixXd::Constant(1,nb_spins,1.0);
  
    // Image kspace
    Eigen::MatrixXcd Mxy = Eigen::MatrixXcd::Zero(nb_meas,nb_lines);

    // Iterate over kspace measurements/kspace points
      for (size_t j = 0; j < nb_lines; j++){
        for (size_t i = 0; i < nb_meas; i++){

        // Fourier exponential
        Eigen::MatrixXcd fe = FourierExp(r, kx(i,j), ky(i,j), kzz);

        // Integral calculation
        // Mxy(i,j) = (ones*(M*Mag.col(2).cwiseProduct(lmbda).cwiseProduct(fe)))[0];
        Mxy(i,j) = (ones*(M*Mag.col(2).cwiseProduct(fe)))[0];

        // Adds the T2 decay
        Mxy(i,j) *= std::exp(-t(i,j)/T2);

      }
    }

    return Mxy;
}



PYBIND11_MODULE(FlowToImage, m) {
    m.doc() = "Utilities for 4d flow image generation";
    m.def("FlowImage3D", &FlowImage3D);
}
