#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
typedef std::vector<Eigen::VectorXcd> ImageVector;
typedef std::tuple<ImageVector, Eigen::VectorXd> ImageTuple;
typedef std::tuple<Eigen::MatrixXcd,
                   Eigen::MatrixXcd,
                   Eigen::MatrixXd> Magnetization_Images;
typedef std::tuple<Eigen::MatrixXcd,
                   Eigen::MatrixXd> Exact_Images;

// Calculates the pixel intensity of the images based on the pixel-to-spins
// connectivity. The intensity is estimated using the mean of all the spins
// inside a given pixel
ImageTuple get_images(const std::vector<Eigen::MatrixXcd> &I,
                    const Eigen::MatrixXd &x,
                    const std::vector<Eigen::VectorXd> &xi,
                    const Eigen::VectorXd &voxel_size,
                    const std::vector<std::vector<int>> &p2s){

    // Number of voxels
    const size_t nr_voxels = p2s.size();

    // Number of input images and encoding directions
    const size_t nr_im = I.size();
    const size_t nr_enc = I[0].cols();

    // Output
    ImageVector Im(nr_im, Eigen::VectorXcd::Zero(nr_voxels*nr_enc));
    Eigen::VectorXd mask = Eigen::VectorXd::Zero(nr_voxels);

    // Iterators and distance
    size_t i, j, k, l;
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
            // for (j=0; j<p2s[i].size(); j++){
            //     w(j) = std::pow(x(p2s[i][j],0) - xi[0](i), 2)
            //          + std::pow(x(p2s[i][j],1) - xi[1](i), 2)
            //          + std::pow(x(p2s[i][j],2) - xi[2](i), 2);
            // }
            // w.array() = (-w.array()/std::pow(d, 2)).exp();

            // Fill images
            for (k=0; k<nr_im; k++){
                for (l=0; l<nr_enc; l++){
                    Im[k](i+l*nr_voxels) = w.cwiseProduct(I[k](p2s[i],l)).sum();
                    // Im[k](i+l*nr_voxels) = I[k](p2s[i],l).mean();
                }
            }
            mask(i) += p2s[i].size();
        }
    }

    return std::make_tuple(Im, mask);

}


// Evaluate the CSPAMM magnetizations on all the spins
Magnetization_Images CSPAMM_magnetizations(const double &M,
                                    const double &M0,
                                    const double &alpha,
                                    const double &beta,
                                    const double &prod,
                                    const double &t,
                                    const double &T1,
                                    const Eigen::VectorXd &ke,
                                    const Eigen::MatrixXd &X){

    // Output images
    Eigen::MatrixXcd Mxy0 = Eigen::MatrixXcd::Zero(X.rows(),X.cols());
    Eigen::MatrixXcd Mxy1 = Eigen::MatrixXcd::Zero(X.rows(),X.cols());;
    Eigen::MatrixXd Mask = Eigen::MatrixXd::Zero(X.rows(),X.cols());;

    // Temp variables
    Eigen::MatrixXcd tmp1 = Eigen::MatrixXcd::Zero(X.rows(),X.cols());
    Eigen::MatrixXcd tmp2 = Eigen::MatrixXcd::Zero(X.rows(),X.cols());

    // Complex unit
    std::complex<double> cu(0, 1);
    double sin2b = std::pow(std::sin(beta),2);
    double cos2b = std::pow(std::cos(beta),2);

    // Build magnetizations
    for (int i=0; i<ke.size(); i++){
        tmp1.array()(Eigen::all,i) += 0.5*M*sin2b*std::exp(-t/T1)*(-cu*ke(i)*X(Eigen::all,i)).array().exp();
        tmp1.array()(Eigen::all,i) += 0.5*M*sin2b*std::exp(-t/T1)*(+cu*ke(i)*X(Eigen::all,i)).array().exp();
        tmp2.array()(Eigen::all,i) += M0*(1.0-std::exp(-t/T1)) + M*cos2b*std::exp(-t/T1);
    }
    Mxy0 += (tmp1 + tmp2)*prod*std::sin(alpha);
    Mxy1 += (-tmp1 + tmp2)*prod*std::sin(alpha);
    Mask.array() += 1.0;

    return std::make_tuple(Mxy0, Mxy1, Mask);

}


// Evaluate the DENSE magnetizations on all the spins
Magnetization_Images DENSE_magnetizations(const double &M,
                                    const double &M0,
                                    const double &alpha,
                                    const double &prod,
                                    const double &t,
                                    const double &T1,
                                    const Eigen::VectorXd &ke,
                                    const Eigen::MatrixXd &X,
                                    const Eigen::MatrixXd &u){

    // Output images
    Eigen::MatrixXcd Mxy0 = Eigen::MatrixXcd::Zero(X.rows(),X.cols());
    Eigen::MatrixXcd Mxy1 = Eigen::MatrixXcd::Zero(X.rows(),X.cols());;
    Eigen::MatrixXd Mask = Eigen::MatrixXd::Zero(X.rows(),X.cols());;

    // Temp variables
    Eigen::MatrixXcd tmp1 = Eigen::MatrixXcd::Zero(X.rows(),X.cols());
    Eigen::MatrixXcd tmp2 = Eigen::MatrixXcd::Zero(X.rows(),X.cols());

    // Complex unit
    std::complex<double> cu(0, 1);

    // Build magnetizations
    for (int i=0; i<ke.size(); i++){
        tmp1.array()(Eigen::all,i) += 0.5*M*std::exp(-t/T1)*(-cu*ke(i)*u(Eigen::all,i)).array().exp();
        tmp1.array()(Eigen::all,i) += 0.5*M*std::exp(-t/T1)*(-cu*ke(i)*(2*X(Eigen::all,i) + u(Eigen::all,i)).array()).exp();
        tmp2.array()(Eigen::all,i) += M0*(1-std::exp(-t/T1))*(-cu*ke(i)*(X(Eigen::all,i) + u(Eigen::all,i)).array()).exp();
    }
    Mxy0 += (tmp1 + tmp2)*prod*std::sin(alpha);
    Mxy1 += (-tmp1 + tmp2)*prod*std::sin(alpha);
    Mask.array() += 1;

    return std::make_tuple(Mxy0, Mxy1, Mask);

}


// Exact magnetization
Exact_Images EXACT_magnetizations(const Eigen::VectorXd &ke,
                                    const Eigen::MatrixXd &u){

    // Output images
    Eigen::MatrixXcd Mxy = Eigen::MatrixXcd::Zero(u.rows(),u.cols());
    Eigen::MatrixXd Mask = Eigen::MatrixXd::Zero(u.rows(),u.cols());

    // Complex unit
    std::complex<double> cu(0, 1);

    // Build magnetizations
    for (int i=0; i<ke.size(); i++){
        Mxy.array()(Eigen::all,i) += (-cu*ke(i)*u(Eigen::all,i)).array().exp();
    }
    Mask.array() += 1;

    return std::make_tuple(Mxy,Mask);

}



PYBIND11_MODULE(ImageBuilding, m) {
    m.doc() = "Utilities for spins-based image generation"; // optional module docstring
    m.def("get_images", &get_images, py::return_value_policy::reference,
      "Calculates the pixel intensity based on the pixel-to-spins connectivity");
    m.def("CSPAMM_magnetizations", &CSPAMM_magnetizations, py::return_value_policy::reference);
    m.def("DENSE_magnetizations", &DENSE_magnetizations, py::return_value_policy::reference);
    m.def("EXACT_magnetizations", &EXACT_magnetizations, py::return_value_policy::reference);
}
