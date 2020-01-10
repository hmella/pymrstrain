#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// BoundingPixels initialization
// This functions builds the map between pixels and voxels
// creating a vector of vectors containing the spins inside
// a specific voxel
std::vector<std::vector<int>> getConnectivity(const Eigen::MatrixXd &x,
                        const std::vector<Eigen::VectorXd> &x_i,
                        const std::vector<int> &local_voxels,
                        const std::vector<double> &voxel_size,
                        const size_t &local_size,
                        const size_t &size,
                        const size_t &slice) {

  // Number of DOFs in ventricle
  const size_t M = x.rows();

  // Vector for group of voxels
  std::vector<std::vector<int>> V(local_size);

  // Vector for individual voxel
  std::vector<int> v;

  // Positions, counter and iterators
  // (OBS: indices i and j represents row and col positions. However,
  //       these indices are flipped with respect to the spatial
  //       coordinates of the image 'x_i'. This mean that 'i' is varying
  //       across the y-coordinate and 'j' across x-coordinate.
  size_t i, k;

  // Voxel half-width
  const double dx = 0.5*voxel_size[0];
  const double dy = 0.5*voxel_size[1];

  py::print(x);
  py::print();

  // Iterates over voxels to perform logical operations
  for (i=0; i<local_size; i++) {

    // Loop over ventricle dofs (Is inside?)
    for (k=0; k<M; k++) {

      // Stores the spins that are inside of the pixel
      if (std::abs(x(k,0) - x_i[0](i)) <= dx && std::abs(x(k,1) - x_i[1](i)) <= dy){
        v.push_back(k);
      }

    }

    // Store each pixel in the global list
    V[i] = v;
    v.clear();

  }

  return V;
}

// ////////////////////////////////////////
// std::vector<int> getConnectivity(const Eigen::MatrixXd &x,
//                         const std::vector<Eigen::VectorXd> &x_i,
//                         const std::vector<int> &local_voxels,
//                         const std::vector<double> &voxel_size,
//                         const size_t &local_size,
//                         const size_t &size,
//                         const size_t &slice) {
//
//   // Number of DOFs in ventricle
//   const size_t M = x.rows();
//
//   // Vector for group of voxels
//   std::vector<int> V(M);
//
//   // Positions, counter and iterators
//   // (OBS: indices i and j represents row and col positions. However,
//   //       these indices are flipped with respect to the spatial
//   //       coordinates of the image 'x_i'. This mean that 'i' is varying
//   //       across the y-coordinate and 'j' across x-coordinate.
//   size_t i, k;
//
//   // Voxel half-width
//   const double dx = 0.5*voxel_size[0];
//   const double dy = 0.5*voxel_size[1];
//
//   // Iterates over voxels to perform logical operations
//   for (k=0; k<M; k++) {
//     // Loop over ventricle dofs (Is inside?)
//     for (i=0; i<local_size; i++) {
//
//       // Stores the spins that are inside of the pixel
//       if ((std::abs(x(k,0) - x_i[0](i)) <= dx) &&
//           (std::abs(x(k,1) - x_i[1](i)) <= dy)){
//         V[k] = i;
//         break;
//       }
//     }
//   }
//
//   return V;
// }








// Get weights
Eigen::VectorXd getWeights(const Eigen::MatrixXd &x,
                        const std::vector<Eigen::VectorXd> &x_i,
                        const std::vector<int> &local_voxels,
                        const std::vector<double> &voxel_size,
                        const size_t &local_size,
                        const size_t &size,
                        const size_t &slice) {

  // Number of DOFs in ventricle
  const size_t M = x.rows();

  // Create image array
  Eigen::VectorXd weights(Eigen::VectorXd::Constant(size, 1.0e-200));

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
  for (i=0; i<local_size; i++) {

    // Loop over ventricle dofs (Is inside?)
    for (k=0; k<M; k++) {
      if (std::abs(x(k,0) - x_i[0](i)) >= dx || std::abs(x(k,1) - x_i[1](i)) >= dy) {}
      else {
        weights(local_voxels[i]) += 1;
      }
    }
  }

  return weights;
}

// Get weights
Eigen::VectorXd getMask(const Eigen::MatrixXd &x,
                        const std::vector<Eigen::VectorXd> &x_i,
                        const std::vector<int> &local_voxels,
                        const std::vector<double> &voxel_size,
                        const size_t &local_size,
                        const size_t &size,
                        const size_t &slice) {

  // Number of DOFs in ventricle
  const size_t M = x.rows();

  // Create image array
  Eigen::VectorXd mask(Eigen::VectorXd::Constant(size, 1.0e-200));

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
  for (i=0; i<local_size; i++) {

    // Loop over ventricle dofs (Is inside?)
    for (k=0; k<M; k++) {
      if (std::abs(x(k,0) - x_i[0](i)) >= dx || std::abs(x(k,1) - x_i[1](i)) >= dy) {}
      else {
        mask(local_voxels[i]) = 1;
        break;
      }
    }
  }

  return mask;
}

// Interpolate FEM values to images
std::vector<Eigen::VectorXd> fem2image_vector_mean(Eigen::VectorXi &mask,
  const Eigen::MatrixXd &x,
  const std::vector<Eigen::VectorXd> &x_i,
  const std::vector<double> &voxel_size,
  const Eigen::VectorXi &local_voxels,
  const Eigen::VectorXi &dofs,
  const Eigen::VectorXd &u,
  const size_t &local_size,
  const size_t &size,
  const size_t slice)
  {

  // Number of DOFs in ventricle
  const size_t M = dofs.size();

  // Create image array
  std::vector<Eigen::VectorXd> ui(2, Eigen::VectorXd::Zero(size));

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
  for (i=0; i<local_size; i++) {
    if (mask(local_voxels[i])) {

      // Loop over ventricle dofs (Is inside?)
      for (k=0; k<M; k++) {
        if (std::abs(x(k,0) - x_i[0](i)) >= dx || std::abs(x(k,1) - x_i[1](i)) >= dy) {}
        else {
          ui[0](local_voxels[i]) += u(dofs(k));
          ui[1](local_voxels[i]) += u(dofs(k)+1);
        }
      }
    }
  }

  return ui;
}

// Gaussian interpolatipon function
Eigen::MatrixXd gaussian(const Eigen::MatrixXd &x,
  const double &radius,
  const double &center_0,
  const double &center_1)
{

    // radius
    Eigen::MatrixXd R = ((x.col(0).array()-center_0).pow(2) + (x.col(1).array()-center_1).pow(2)).sqrt();

    // gaussian
    Eigen::MatrixXd f;
    double sigma = radius*radius;
    f = -R.array().pow(2)/(sigma);
    f = f.array().exp();

    return f;

}


// Interpolate FEM values to images
std::vector<Eigen::VectorXd> fem2image_vector_gaussian(Eigen::VectorXi &mask,
  const Eigen::MatrixXd &x,
  const std::vector<Eigen::VectorXd> &x_i,
  const std::vector<double> &voxel_size,
  const Eigen::VectorXi &local_voxels,
  const Eigen::VectorXi &dofs,
  const Eigen::VectorXd &u,
  const size_t &local_size,
  const size_t &size,
  const size_t slice)
  {

  // Sinc interpolator
  Eigen::VectorXd f;

  // Number of DOFs in ventricle
  const size_t M = dofs.size();

  // Create image array Eigen::MatrixXd
  std::vector<Eigen::VectorXd> ui(2, Eigen::VectorXd::Zero(size));

  // Positions, counter and iterators
  // (OBS: indices i and j represents row and col positions. However,
  //       these indices are flipped with respect to the spatial
  //       coordinates of the image 'x_i'. This mean that 'i' is varying
  //       across the y-coordinate and 'j' across x-coordinate.
  size_t i;

  // Voxel half-width
  const double dx = 0.1*voxel_size[0];
  const double dy = 0.1*voxel_size[1];
  const double r  = std::sqrt(std::pow(dx,2)+std::pow(dy,2));

  // Iterates over voxels to perform logical operations
  for (i=0; i<local_size; i++) {
    if (mask(local_voxels[i])) {

      // Sinc interpolator
      f = gaussian(x,r,x_i[0](i),x_i[1](i));
      f /= f.sum();

      // Pixel intensity
      ui[0](local_voxels[i]) = (u(Eigen::seq(0,M-1,2),Eigen::all)).array().cwiseProduct(f.array()).sum();
      ui[1](local_voxels[i]) = (u(Eigen::seq(1,M-1,2),Eigen::all)).array().cwiseProduct(f.array()).sum();

    }
  }

  return ui;
}


// Interpolate FEM values to images (special case: SPAMM)
std::vector<Eigen::MatrixXd> fem2image_scalar(Eigen::VectorXi &mask,
                                              const Eigen::MatrixXd &x,
                                              const std::vector<Eigen::VectorXd> &x_i,
                                              const std::vector<double> &voxel_size,
                                              const Eigen::VectorXi &dofs,
                                              const Eigen::VectorXd &u,
                                              const std::vector<size_t> &size,
                                              const size_t slice) {

  // Number of DOFs in ventricle
  const size_t M = dofs.size();

  // Create image array
  std::vector<Eigen::MatrixXd> ui(2, Eigen::MatrixXd::Zero(size[0], size[1]));
  ui[1] = Eigen::MatrixXd::Constant(size[0], size[1], 1.0e-200);

  // Positions, counter and iterators
  // (OBS: indices i and j represents row and col positions. However,
  //       these indices are flipped with respect to the spatial
  //       coordinates of the image 'x_i'. This mean that 'i' is varying
  //       across the y-coordinate and 'j' across x-coordinate.
  size_t i, j, k;
  size_t count;

  // Voxel half-width
  const double dx = 0.5*voxel_size[0];
  const double dy = 0.5*voxel_size[1];

  // Iterates over voxels to perform logical operations
  for (i=0; i<size[0]; i++) {
    for (j=0; j<size[1]; j++) {

      // Loop over ventricle dofs (Is inside?)
      count = 0;
      for (k=0; k<M; k++) {
        if (std::abs(x(k,0) - x_i[0](j)) >= dx || std::abs(x(k,1) - x_i[1](i)) >= dy) {}
        else {
          mask(count) = k;
          count += 1;
        }
      }

      // Assign dofs values to image
      for (k=0; k<count; k++) {
        ui[0](i,j) += u(dofs(mask(k)));   // Sum of values
      }

      // Number of values in each pixel
      if (count != 0) {
        ui[1](i,j) = count;
      }

      // Clear mask
      mask = Eigen::VectorXi::Zero(mask.size());
    }
  }

  return ui;
}


// (TODO: FIX 3D VERSION) Interpolate FEM values to images
std::vector<Eigen::MatrixXd> fem2image_vector_3d(Eigen::VectorXi &mask,
                                                 const Eigen::MatrixXd &x,
                                                 const std::vector<Eigen::VectorXd> &x_i,
                                                 const std::vector<double> &voxel_size,
                                                 const Eigen::VectorXi &dofs,
                                                 const Eigen::VectorXd &u,
                                                 const std::vector<size_t> &size,
                                                 const size_t slice) {

  // Number of DOFs in ventricle
  const size_t M = dofs.size();

  // Create image array
  std::vector<Eigen::MatrixXd> ui(4, Eigen::MatrixXd::Zero(size[0], size[1]));
  ui[3] = Eigen::MatrixXd::Constant(size[0], size[1], 1.0e-200);

  // Positions, counter and iterators
  // (OBS: indices i and j represents row and col positions. However,
  //       these indices are flipped with respect to the spatial
  //       coordinates of the image 'x_i'. This mean that 'i' is varying
  //       across the y-coordinate and 'j' across x-coordinate.
  size_t i, j, k;
  size_t count;

  // Voxel half-width
  const double dx = 0.5*voxel_size[0];
  const double dy = 0.5*voxel_size[1];
  const double dz = 0.5*voxel_size[2];

  // Iterates over voxels to perform logical operations
  for (i=0; i<size[0]; i++) {
    for (j=0; j<size[1]; j++) {

      // Loop over ventricle dofs (Is inside?)
      count = 0;
      for (k=0; k<M; k++) {
        if (std::abs(x(k,0) - x_i[0](j)) >= dx || std::abs(x(k,1) - x_i[1](i)) >= dy || std::abs(x(k,2) - x_i[2](slice)) >= dz) {}
        else {
          mask(count) = k;
          count += 1;
        }
      }

      // Assign dofs values to image
      for (k=0; k<count; k++) {
        ui[0](i,j) += u(dofs(mask(k)));
        ui[1](i,j) += u(dofs(mask(k))+1);
        ui[2](i,j) += u(dofs(mask(k))+2);
      }

      // Number of values in each pixel
      if (count != 0) {
        ui[3](i,j) = count;
      }

      // Clear mask
      mask = Eigen::VectorXi::Zero(mask.size());
    }
  }

  return ui;
}


// (TODO: FIX 3D VERSION) Interpolate FEM values to images (special case: SPAMM)
std::vector<Eigen::MatrixXd> fem2image_scalar_3d(Eigen::VectorXi &mask,
                                              const Eigen::MatrixXd &x,
                                              const std::vector<Eigen::VectorXd> &x_i,
                                              const std::vector<double> &voxel_size,
                                              const Eigen::VectorXi &dofs,
                                              const Eigen::VectorXd &u,
                                              const std::vector<size_t> &size,
                                              const size_t slice) {

  // Number of DOFs in ventricle
  const size_t M = dofs.size();

  // Create image array
  std::vector<Eigen::MatrixXd> ui(2, Eigen::MatrixXd::Zero(size[0], size[1]));
  ui[1] = Eigen::MatrixXd::Constant(size[0], size[1], 1.0e-200);

  // Positions, counter and iterators
  // (OBS: indices i and j represents row and col positions. However,
  //       these indices are flipped with respect to the spatial
  //       coordinates of the image 'x_i'. This mean that 'i' is varying
  //       across the y-coordinate and 'j' across x-coordinate.
  size_t i, j, k;
  size_t count;

  // Voxel half-width
  const double dx = 0.5*voxel_size[0];
  const double dy = 0.5*voxel_size[1];
  const double dz = 0.5*voxel_size[2];

  // Iterates over voxels to perform logical operations
  for (i=0; i<size[0]; i++) {
    for (j=0; j<size[1]; j++) {
      // Loop over ventricle dofs (Is inside?)
      count = 0;
      for (k=0; k<M; k++) {
        if (std::abs(x(k,0) - x_i[0](j)) >= dx || std::abs(x(k,1) - x_i[1](i)) >= dy || std::abs(x(k,2) - x_i[2](slice)) >= dz) {}
        else {
          mask(count) = k;
          count += 1;
        }
      }

      // Assign dofs values to image
      for (k=0; k<count; k++) {
        ui[0](i,j) += u(dofs(mask(k)));
      }

      // Number of values in each pixel
      if (count != 0) {
        ui[1](i,j) = count;
      }

      // Clear mask
      mask = Eigen::VectorXi::Zero(mask.size());
    }
  }

  return ui;
}


PYBIND11_MODULE(ImageUtilities, m) {
    m.doc() = "Utilities for Image generation"; // optional module docstring
    m.def("getConnectivity", &getConnectivity, py::return_value_policy::reference, "Get connectivity between spins and voxels");
    m.def("getWeights", &getWeights, py::return_value_policy::reference, "Get weights from FEM data");
    m.def("getMask", &getMask, py::return_value_policy::reference, "Get mask from FEM data");
    m.def("fem2image_scalar", &fem2image_scalar, py::return_value_policy::reference, "Image interpolation for scalar data");
    m.def("fem2image_vector_mean", &fem2image_vector_mean, py::return_value_policy::reference, "Image interpolation for vector data");
    m.def("fem2image_vector_gaussian", &fem2image_vector_gaussian, py::return_value_policy::reference, "Image interpolation for vector data");
    m.def("fem2image_scalar_3d", &fem2image_scalar_3d, py::return_value_policy::reference, "Image interpolation for scalar data (3D version)");
    m.def("fem2image_vector_3d", &fem2image_vector_3d, py::return_value_policy::reference, "Image interpolation for vector data (3D version)");
}
