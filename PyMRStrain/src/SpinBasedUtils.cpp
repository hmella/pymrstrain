#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Connectivity initialization
// This functions builds the map between pixels and voxels
// creating a vector of vectors containing the spins inside
// a specific voxel
std::vector<int> getConnectivity(const Eigen::MatrixXd &x,
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
    if (p2s[i].empty() != 1){
        output[0](i) = u(p2s[i],Eigen::all).mean();
        output[1](i) = 1;
    }
}

return output;

}

// Updates the pixel-to-spin connectivity using the spin-to-pixel vector
std::vector<std::vector<int>> update_p2s(const std::vector<int> s2p,
                                         const size_t nr_voxels){

    // Pixel-to-spins connectivity
    std::vector<std::vector<int>> p2s(nr_voxels);

    // Number of spins
    const size_t nr_spins = s2p.size();

    // Update connectivity
    for (size_t i=0; i<nr_spins; i++){
      p2s[s2p[i]].push_back(i);
    }

    return p2s;
}


PYBIND11_MODULE(SpinBasedutils, m) {
    m.doc() = "Utilities for spins-based image generation"; // optional module docstring
    m.def("getConnectivity", &getConnectivity, py::return_value_policy::reference, "Get connectivity between spins and voxels");
    m.def("getImage", &getImage, py::return_value_policy::reference, "Calculates the pixel intensity based on the pixel-to-spins connectivity");
    m.def("update_p2s", &update_p2s, py::return_value_policy::reference, "Updates the pixel-to-spin connectivity using the spin-to-pixel vector");
}
