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
                        const std::vector<int> &local_voxels,
                        const std::vector<double> &voxel_size,
                        const size_t &local_size,
                        const size_t &size) {

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
    for (i=0; i<local_size; i++) {

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
                        const std::vector<int> &local_voxels,
                        const std::vector<double> &voxel_size,
                        const size_t &local_size,
                        const size_t &size) {

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
  const double dz = 0.5*voxel_size[2];

  // Iterates over voxels to perform logical operations
  for (k=0; k<M; k++) {
    // Loop over ventricle dofs (Is inside?)
    for (i=0; i<local_size; i++) {

      // Stores the spins that are inside of the pixel
      if ((std::abs(x(k,0) - x_i[0](i)) <= dx) &&
          (std::abs(x(k,1) - x_i[1](i)) <= dy) &&
          (std::abs(x(k,2) - x_i[2](i)) <= dz)){
        V[k] = i;
        break;
      }
    }
  }

  return V;
}

// Calculates the pixel intensity of the images based on the pixel-to-spins
// connectivity. The intensity is estimated using the mean of all the spins
// inside a fgiven pixel
std::vector<Eigen::VectorXd> getImage(const Eigen::MatrixXd u,
                          const std::vector<std::vector<int>> p2s){

// Number of pixels
const size_t nr_voxels = p2s.size();

// Output
std::vector<Eigen::VectorXd> output(2);
output[0] = Eigen::VectorXd::Zero(nr_voxels);
output[1] = Eigen::VectorXd::Zero(nr_voxels);

// Number of pixels
for (size_t i=0; i<nr_voxels; i++){
    if (!p2s[i].empty()){
        output[0](i) = u(p2s[i],Eigen::all).mean();
        output[1](i) = 1;
    }
}

return output;

}

// Updates the pixel-to-spin connectivity using the spin-to-pixel vector
std::tuple<std::vector<std::vector<int>>, std::vector<int>> update_p2s(const std::vector<int> s2p,
                                         const int nr_voxels){

    // Pixel-to-spins connectivity
    std::vector<std::vector<int>> p2s(nr_voxels);

    // Number of spins
    const size_t nr_spins = s2p.size();

    // Signal weights
    std::vector<int> signal_w(nr_voxels);

    // Update connectivity
    // If the spins go outside of its container voxel
    // then they are not included in the pixel-to-spins
    // connectivity
    for (size_t i=0; i<nr_spins; i++){
        if ((s2p[i] >= 0) && (s2p[i] < nr_voxels)){
            p2s[s2p[i]].push_back(i);
            signal_w[s2p[i]] += 1;
        }
    }

    return std::make_tuple(p2s, signal_w);
}

PYBIND11_MODULE(SpinBasedUtils, m) {
    m.doc() = "Utilities for spins-based image generation"; // optional module docstring
    m.def("getConnectivity2", &getConnectivity2, py::return_value_policy::reference, "Get connectivity between spins and voxels");
    m.def("getConnectivity3", &getConnectivity3, py::return_value_policy::reference, "Get connectivity between spins and voxels for 3D images");
    m.def("getImage", &getImage, py::return_value_policy::reference, "Calculates the pixel intensity based on the pixel-to-spins connectivity");
    m.def("update_p2s", &update_p2s, py::return_value_policy::reference, "Updates the pixel-to-spin connectivity using the spin-to-pixel vector");
}
