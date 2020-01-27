#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Connectivity initialization
// This functions builds the map between pixels and voxels
// creating a vector of vectors containing the spins inside
// a specific voxel
std::vector<int> getConnectivity2(const Eigen::MatrixXd &x,
                        const std::vector<Eigen::VectorXd> &x_i,
                        const std::vector<double> &voxel_size,
                        const size_t &nr_voxels) {

  // Number of DOFs in ventricle
  const size_t M = x.rows();

  // Vector for group of voxels
  std::vector<int> V(M);

  // Positions, counter and iterators
  // (OBS: indices i and j represents row and col positions. However,
  //       these indices are flipped with respect to the spatial
  //       coordinates of the image 'x_i'. This mean that 'i' is varying
  //       across the y-coordinate and 'j' across x-coordinate.
  size_t i, k;

  // Voxel half-width
  const double dx = 0.5*voxel_size[0];
  const double dy = 0.5*voxel_size[1];

  // Iterates over voxels to perform logical operations
  for (k=0; k<M; k++) {
    // Loop over ventricle dofs (Is inside?)
    for (i=0; i<nr_voxels; i++) {

      // Stores the spins that are inside of the pixel
      if ((std::abs(x(k,0) - x_i[0](i)) <= dx) &&
          (std::abs(x(k,1) - x_i[1](i)) <= dy)){
        V[k] = i;
        break;
      }
    }
  }

  return V;
}

// Connectivity initialization for 3D images
// This functions builds the map between pixels and voxels
// creating a vector of vectors containing the spins inside
// a specific voxel
typedef std::tuple<std::vector<int>, std::vector<int>> ConnectivityVectors;
ConnectivityVectors getConnectivity3(const Eigen::MatrixXf &x,
                        const std::vector<Eigen::VectorXf> &x_i,
                        const std::vector<float> &voxel_size) {

  // Number of spins and voxels
  const int nr_spins = x.rows();
  const int nr_voxels = x_i[0].size();

  // Vector for spin-to-pixel connectivity
  // std::vector<int> s2p(nr_spins, -1);
  std::vector<int> s2p;
  std::vector<int> excited_spins;

  // Positions, counter and iterators
  // (OBS: indices i and j represents row and col positions. However,
  //       these indices are flipped with respect to the spatial
  //       coordinates of the image 'x_i'. This mean that 'i' is varying
  //       across the y-coordinate and 'j' across x-coordinate.
  int voxel, spin;

  // Voxel half-width
  const float dx = 0.5*voxel_size[0];
  const float dy = 0.5*voxel_size[1];
  const float dz = 0.5*voxel_size[2];

  // Iterates over voxels to perform logical operations
  for (spin=0; spin<nr_spins; spin++) {

      // Loop over voxels (Is inside?)
      for (voxel=0; voxel<nr_voxels; voxel++) {

          // Stores the spins that are inside of the pixel
          if (((x(spin,0) - x_i[0](voxel)) > dx && (x(spin,0) - x_i[0](voxel)) < -dx) ||
              ((x(spin,1) - x_i[1](voxel)) > dy && (x(spin,1) - x_i[1](voxel)) < -dy) ||
              ((x(spin,2) - x_i[2](voxel)) > dz && (x(spin,2) - x_i[2](voxel)) < -dz)){
          // if (((x(spin,0) - x_i[0](voxel)) > dx || (x(spin,0) - x_i[0](voxel)) < -dx) ||
          //     ((x(spin,1) - x_i[1](voxel)) > dy || (x(spin,1) - x_i[1](voxel)) < -dy) ||
          //     ((x(spin,2) - x_i[2](voxel)) > dz || (x(spin,2) - x_i[2](voxel)) < -dz)){
              // Do nothing
          }
          else{
              // s2p[k] = i;
              s2p.push_back(voxel);
              excited_spins.push_back(spin);
              break;
          }

      }

  }

  return std::make_tuple(s2p, excited_spins);
}

// Updates the pixel-to-spin connectivity using the spin-to-pixel vector
std::vector<std::vector<int>> update_p2s(const std::vector<int> &s2p,
                                         const std::vector<int> &excited_spins,
                                         const int &nr_voxels){

    // Pixel-to-spins connectivity
    std::vector<std::vector<int>> p2s(nr_voxels);

    // Number of spins
    const size_t nr_spins = s2p.size();

    // Update connectivity
    // If the spins go outside of its container voxel
    // then they are not included in the pixel-to-spins
    // connectivity
    for (size_t i=0; i<nr_spins; i++){
        if ((s2p[i] >= 0) && (s2p[i] < nr_voxels)){
            p2s[s2p[i]].push_back(excited_spins[i]);
        }
    }

    return p2s;
}


// Calculates the pixel intensity of the images based on the pixel-to-spins
// connectivity. The intensity is estimated using the mean of all the spins
// inside a given pixel
typedef std::vector<Eigen::VectorXcf> ImageVector;
typedef std::tuple<ImageVector, Eigen::VectorXf> ImageTuple;
ImageTuple getImage(const std::vector<Eigen::VectorXcf> &I,
                    const Eigen::MatrixXf &x,
                    const std::vector<Eigen::VectorXf> &xi,
                    const Eigen::VectorXf &voxel_size,
                    const std::vector<std::vector<int>> &p2s){

    // Number of voxels
    const size_t nr_voxels = p2s.size();

    // Number of input images
    const size_t nr_im = I.size();

    // Output
    ImageVector Im(nr_im, Eigen::VectorXcf::Zero(nr_voxels));
    Eigen::VectorXf mask = Eigen::VectorXf::Zero(nr_voxels);

    // Iterators and distance
    size_t i, j, k;
    double d = 0.707106781*voxel_size.norm();

    // Number of pixels
    for (i=0; i<nr_voxels; i++){
        if (!p2s[i].empty()){

            // Build weigths
            Eigen::VectorXf w = Eigen::VectorXf::Zero(p2s[i].size());
            for (j=0; j<p2s[i].size(); j++){
                w(j) = std::pow(std::pow(x(p2s[i][j],0) - xi[0](i), 2)
                     + std::pow(x(p2s[i][j],1) - xi[1](i), 2)
                     + std::pow(x(p2s[i][j],2) - xi[2](i), 2), 0.5);
            }
            w.array() = 1.0 - w.array()/d;

            // Fill images
            for (k=0; k<nr_im; k++){
                Im[k](i) = w.cwiseProduct(I[k](p2s[i])).sum();
            }
            mask(i) += p2s[i].size();
        }
    }

return std::make_tuple(Im, mask);

}


ImageTuple getImage_(const std::vector<Eigen::MatrixXcf> &I,
                    const Eigen::MatrixXf &x,
                    const std::vector<Eigen::VectorXf> &xi,
                    const Eigen::VectorXf &voxel_size,
                    const std::vector<std::vector<int>> &p2s){

    // Number of voxels
    const size_t nr_voxels = p2s.size();

    // Number of input images and encoding directions
    const size_t nr_im = I.size();
    const size_t nr_enc = I[0].cols();

    // Output
    ImageVector Im(nr_im, Eigen::VectorXcf::Zero(nr_voxels*nr_enc));
    Eigen::VectorXf mask = Eigen::VectorXf::Zero(nr_voxels);

    // Iterators and distance
    size_t i, j, k, l;
    double d = 0.707106781*voxel_size.norm();

    // Number of pixels
    for (i=0; i<nr_voxels; i++){
        if (!p2s[i].empty()){

            // Build weigths
            Eigen::VectorXf w = Eigen::VectorXf::Zero(p2s[i].size());
            for (j=0; j<p2s[i].size(); j++){
                w(j) = std::pow(std::pow(x(p2s[i][j],0) - xi[0](i), 2)
                     + std::pow(x(p2s[i][j],1) - xi[1](i), 2)
                     + std::pow(x(p2s[i][j],2) - xi[2](i), 2), 0.5);
            }
            w.array() = 1.0 - w.array()/d;

            // Fill images
            for (k=0; k<nr_im; k++){
                for (l=0; l<nr_enc; l++){
                    Im[k](i+l*nr_voxels) = w.cwiseProduct(I[k](p2s[i],l)).sum();
                }
            }
            mask(i) += p2s[i].size();
        }
    }

    return std::make_tuple(Im, mask);

}

// Evaluate the DENSE magnetizations on all the spins
typedef std::tuple<Eigen::MatrixXcf, Eigen::MatrixXcf, Eigen::MatrixXcf> Magnetization_Images;
Magnetization_Images magnetizations(const float &M,
                                    const float &M0,
                                    const float &alpha,
                                    const float &prod,
                                    const float &t,
                                    const float &T1,
                                    const Eigen::VectorXf &ke,
                                    const Eigen::MatrixXf &X,
                                    const Eigen::MatrixXf &u){

    // Output images
    Eigen::MatrixXcf Mxy0 = Eigen::MatrixXcf::Zero(X.rows(),X.cols());
    Eigen::MatrixXcf Mxy1 = Eigen::MatrixXcf::Zero(X.rows(),X.cols());;
    // Eigen::MatrixXcd Mxyin = Eigen::MatrixXcd::Zero(X.rows(),X.cols());;

    // Temp variables
    Eigen::MatrixXcf tmp1 = Eigen::MatrixXcf::Zero(X.rows(),X.cols());
    Eigen::MatrixXcf tmp2 = Eigen::MatrixXcf::Zero(X.rows(),X.cols());

    // Complex unit
    std::complex<float> cu(0, 1);

    // Build magnetizations
    for (int i=0; i<ke.size(); i++){
        tmp1.array()(Eigen::all,i) += 0.5*M*std::exp(-t/T1)*(-cu*ke(i)*u(Eigen::all,i)).array().exp();
        tmp1.array()(Eigen::all,i) += 0.5*M*std::exp(-t/T1)*(-cu*ke(i)*(2*X(Eigen::all,i) + u(Eigen::all,i))).array().exp();
        tmp2.array()(Eigen::all,i) += M0*(1-std::exp(-t/T1))*(-cu*ke(i)*(X(Eigen::all,i) + u(Eigen::all,i))).array().exp();
    }
    Mxy0 += (tmp1 + tmp2)*prod*std::sin(alpha);
    Mxy1 += (-tmp1 + tmp2)*prod*std::sin(alpha);

    return std::make_tuple(Mxy0, Mxy1, Mxy0);

}


PYBIND11_MODULE(SpinBasedUtils, m) {
    m.doc() = "Utilities for spins-based image generation"; // optional module docstring
    m.def("getConnectivity2", &getConnectivity2, py::return_value_policy::reference, "Get connectivity between spins and voxels");
    m.def("getConnectivity3", &getConnectivity3, py::return_value_policy::reference, "Get connectivity between spins and voxels for 3D images");
    m.def("getImage", &getImage, py::return_value_policy::reference, "Calculates the pixel intensity based on the pixel-to-spins connectivity");
    m.def("getImage_", &getImage_, py::return_value_policy::reference, "Calculates the pixel intensity based on the pixel-to-spins connectivity");
    m.def("update_p2s", &update_p2s, py::return_value_policy::reference, "Updates the pixel-to-spin connectivity using the spin-to-pixel vector");
    m.def("magnetizations", &magnetizations, py::return_value_policy::reference);
}
