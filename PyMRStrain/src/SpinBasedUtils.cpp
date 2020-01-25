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
std::vector<int> getConnectivity3(const Eigen::MatrixXd &x,
                        const std::vector<Eigen::VectorXd> &x_i,
                        const std::vector<double> &voxel_size) {

  // Number of spins and voxels
  const int nr_spins = x.rows();
  const int nr_voxels = x_i[0].size();

  // Vector for spin-to-pixel connectivity
  std::vector<int> s2p(nr_spins);
  std::fill(s2p.begin(), s2p.end(), -1);

  // Positions, counter and iterators
  // (OBS: indices i and j represents row and col positions. However,
  //       these indices are flipped with respect to the spatial
  //       coordinates of the image 'x_i'. This mean that 'i' is varying
  //       across the y-coordinate and 'j' across x-coordinate.
  int i, k;

  // Voxel half-width
  const double dx = 0.5*voxel_size[0];
  const double dy = 0.5*voxel_size[1];
  const double dz = 0.5*voxel_size[2];

  // Iterates over voxels to perform logical operations
  for (k=0; k<nr_spins; k++) {

      // Loop over voxels (Is inside?)
      for (i=0; i<nr_voxels; i++) {

          // Stores the spins that are inside of the pixel
          if ((std::abs(x(k,0) - x_i[0](i)) > dx) ||
              (std::abs(x(k,1) - x_i[1](i)) > dy) ||
              (std::abs(x(k,2) - x_i[2](i)) > dz)){
              // Do nothing
          }
          else{
              s2p[k] = i;
              break;
          }

      }

  }

  return s2p;
}

// Updates the pixel-to-spin connectivity using the spin-to-pixel vector
std::vector<std::vector<int>> update_p2s(const std::vector<int> s2p,
                                         const int nr_voxels){

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
            p2s[s2p[i]].push_back(i);
        }
    }

    return p2s;
}


// Calculates the pixel intensity of the images based on the pixel-to-spins
// connectivity. The intensity is estimated using the mean of all the spins
// inside a given pixel
typedef std::vector<Eigen::VectorXcf> ImageVector;
typedef std::tuple<ImageVector, Eigen::VectorXi> ImageTuple;
ImageTuple getImage(const std::vector<Eigen::VectorXcf> &I,
                    const Eigen::MatrixXd &x,
                    const std::vector<Eigen::VectorXd> &xi,
                    const Eigen::VectorXd &voxel_size,
                    const std::vector<std::vector<int>> &p2s){

    // Number of voxels
    const size_t nr_voxels = p2s.size();

    // Number of input images
    const size_t nr_im = I.size();

    // Output
    ImageVector Im(nr_im, Eigen::VectorXcf::Zero(nr_voxels));
    Eigen::VectorXi mask = Eigen::VectorXi::Zero(nr_voxels);

    // Iterators and distance
    size_t i, j, k;
    double d = 0.707106781*voxel_size.norm();

    // Number of pixels
    for (i=0; i<nr_voxels; i++){
        if (!p2s[i].empty()){

            // Build weigths
            Eigen::VectorXd w = Eigen::VectorXd::Zero(p2s[i].size());
            for (j=0; j<p2s[i].size(); j++){
                w(j) = std::pow(std::pow(x(p2s[i][j],0) - xi[0](i), 2)
                     + std::pow(x(p2s[i][j],1) - xi[1](i), 2)
                     + std::pow(x(p2s[i][j],2) - xi[2](i), 2), 0.5);
            }
            w.array() = 1.0 - w.array()/d;

            // Fill images
            for (k=0; k<nr_im; k++){
                Im[k](i) = (w.cast<std::complex<float>>().cwiseProduct(I[k](p2s[i]))).sum();
            }
            mask(i) += 1;
        }
    }

return std::make_tuple(Im, mask);

}


PYBIND11_MODULE(SpinBasedUtils, m) {
    m.doc() = "Utilities for spins-based image generation"; // optional module docstring
    m.def("getConnectivity2", &getConnectivity2, py::return_value_policy::reference, "Get connectivity between spins and voxels");
    m.def("getConnectivity3", &getConnectivity3, py::return_value_policy::reference, "Get connectivity between spins and voxels for 3D images");
    m.def("getImage", &getImage, py::return_value_policy::reference, "Calculates the pixel intensity based on the pixel-to-spins connectivity");
    m.def("update_p2s", &update_p2s, py::return_value_policy::reference, "Updates the pixel-to-spin connectivity using the spin-to-pixel vector");
}
