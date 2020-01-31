#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
typedef std::vector<Eigen::VectorXcf> ImageVector;
typedef std::tuple<ImageVector, Eigen::VectorXf> ImageTuple;
typedef std::tuple<Eigen::MatrixXcf,
                   Eigen::MatrixXcf,
                   Eigen::MatrixXcf> Magnetization_Images;


// Calculates the pixel intensity of the images based on the pixel-to-spins
// connectivity. The intensity is estimated using the mean of all the spins
// inside a given pixel
ImageTuple get_images(const std::vector<Eigen::MatrixXcf> &I,
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
Magnetization_Images DENSE_magnetizations(const float &M,
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


// Evaluate the CSPAMM magnetizations on all the spins
Magnetization_Images CSPAMM_magnetizations(const float &M,
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
        tmp1.array()(Eigen::all,i) += 0.5*M*std::exp(-t/T1)*(-cu*ke(i)*X(Eigen::all,i)).array().exp();
        tmp1.array()(Eigen::all,i) += 0.5*M*std::exp(-t/T1)*(+cu*ke(i)*X(Eigen::all,i)).array().exp();
        tmp2.array()(Eigen::all,i) += M0*(1-std::exp(-t/T1));
    }
    Mxy0 += (tmp1 + tmp2)*prod*std::sin(alpha);
    Mxy1 += (-tmp1 + tmp2)*prod*std::sin(alpha);

    return std::make_tuple(Mxy0, Mxy1, Mxy0);

}

PYBIND11_MODULE(ImageBuilding, m) {
    m.doc() = "Utilities for spins-based image generation"; // optional module docstring
    m.def("get_images", &get_images, py::return_value_policy::reference,
      "Calculates the pixel intensity based on the pixel-to-spins connectivity");
    m.def("DENSE_magnetizations", &DENSE_magnetizations, py::return_value_policy::reference);
    m.def("CSPAMM_magnetizations", &CSPAMM_magnetizations, py::return_value_policy::reference);
}
