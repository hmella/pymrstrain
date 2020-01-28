#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
typedef std::tuple<std::vector<int>, std::vector<int>> ConnectivityVectors;

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

PYBIND11_MODULE(Connectivity, m) {
		m.def("getConnectivity2", &getConnectivity2,
			py::return_value_policy::reference, "Build spins-to-voxels connectivity");
		m.def("getConnectivity3", &getConnectivity3,
			py::return_value_policy::reference, "Build spins-to-voxels connectivity");
		m.def("update_p2s", &update_p2s,
			py::return_value_policy::reference, "Updates the pixel-to-spins	connectivity");
}
