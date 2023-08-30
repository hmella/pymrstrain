#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace Eigen;
typedef std::vector<VectorXcd> ImageVector;
typedef std::tuple<ImageVector, VectorXd> ImageTuple;
typedef std::tuple<MatrixXcd,
                   MatrixXcd,
                   MatrixXd> Magnetization_Images;

typedef std::tuple<MatrixXcd,
                   MatrixXcd,
                   MatrixXd,
                   MatrixXcd> Magnetization_Images_T1;                   
typedef std::tuple<MatrixXcd,
                   MatrixXd> Exact_Images;

// Calculates the pixel intensity of the images based on the pixel-to-spins
// connectivity. The intensity is estimated using the mean of all the spins
// inside a given pixel
ImageTuple get_images(const std::vector<MatrixXcd> &I,
                    const MatrixXd &x,
                    const std::vector<VectorXd> &xi,
                    const VectorXd &voxel_size,
                    const std::vector<std::vector<int>> &p2s){

    // Number of voxels
    const size_t nr_voxels = p2s.size();

    // Number of input images and encoding directions
    const size_t nr_im = I.size();
    const size_t nr_enc = I[0].cols();

    // Output
    ImageVector Im(nr_im, VectorXcd::Zero(nr_voxels*nr_enc));
    VectorXd mask = VectorXd::Zero(nr_voxels);

    // Iterators and distance
    size_t i, j, k, l;
    double d = 0.707106781*voxel_size.norm();

    // Number of pixels
    for (i=0; i<nr_voxels; i++){
        if (!p2s[i].empty()){

            // Build weigths
            VectorXd w = VectorXd::Zero(p2s[i].size());
            for (j=0; j<p2s[i].size(); j++){
                // w(j) = std::pow(std::pow(x(p2s[i][j],0) - xi[0](i), 2)
                //      + std::pow(x(p2s[i][j],1) - xi[1](i), 2)
                //      + std::pow(x(p2s[i][j],2) - xi[2](i), 2), 0.5);
                w(j) = std::pow(std::pow(x(p2s[i][j],0) - xi[0](i), 2)
                              + std::pow(x(p2s[i][j],1) - xi[1](i), 2), 0.5);
            }
            w.array() = 1.0 - w.array()/d;
            // for (j=0; j<p2s[i].size(); j++){
            //     // w(j) = std::pow(x(p2s[i][j],0) - xi[0](i), 2)
            //     //      + std::pow(x(p2s[i][j],1) - xi[1](i), 2)
            //     //      + std::pow(x(p2s[i][j],2) - xi[2](i), 2);
            //     w(j) = std::pow(x(p2s[i][j],0) - xi[0](i), 2)
            //          + std::pow(x(p2s[i][j],1) - xi[1](i), 2);
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
Magnetization_Images CSPAMM_magnetizations(const double &alpha,
                                    const double &beta,
                                    const double &prod,
                                    const VectorXd &M0,
                                    const VectorXd &exp_T1,
                                    const VectorXd &ke,
                                    const MatrixXd &X){

    // Output images
    MatrixXcd Mxy0 = MatrixXcd::Zero(X.rows(),X.cols());
    MatrixXcd Mxy1 = MatrixXcd::Zero(X.rows(),X.cols());;
    MatrixXd Mask = MatrixXd::Zero(X.rows(),X.cols());;

    // Temp variables
    MatrixXcd tmp1 = MatrixXcd::Zero(X.rows(),X.cols());
    MatrixXcd tmp2 = MatrixXcd::Zero(X.rows(),X.cols());

    // Complex unit
    std::complex<double> cu(0, 1);
    double sin2b = std::pow(std::sin(beta),2);
    double cos2b = std::pow(std::cos(beta),2);

    // Build magnetizations
    for (int i=0; i<ke.size(); i++){
        tmp1.array()(indexing::all,i) += 0.5*M0.array()*sin2b*exp_T1.array()*(-cu*ke(i)*X(indexing::all,i)).array().exp();
        tmp1.array()(indexing::all,i) += 0.5*M0.array()*sin2b*exp_T1.array()*(+cu*ke(i)*X(indexing::all,i)).array().exp();
        tmp2.array()(indexing::all,i) += M0.array()*(1.0-exp_T1.array()) + M0.array()*cos2b*exp_T1.array();
    }
    Mxy0 += (tmp1 + tmp2)*prod*std::sin(alpha);
    Mxy1 += (-tmp1 + tmp2)*prod*std::sin(alpha);
    Mask.array() += 1.0;

    return std::make_tuple(Mxy0, Mxy1, Mask);

}


// Evaluate the ORI-O-CSPAMM magnetizations on all the spins
Magnetization_Images_T1 ORI_O_CSPAMM_magnetizations(const double &alpha,
                                    const double &beta,
                                    const double &prod,
                                    const VectorXd &M0,
                                    const VectorXd &exp_T1,
                                    const VectorXd &T1,
                                    const VectorXd &ke,
                                    const MatrixXd &X){

    // Output images
    MatrixXcd Mxy0 = MatrixXcd::Zero(X.rows(),1);
    MatrixXcd Mxy1 = MatrixXcd::Zero(X.rows(),1);
    MatrixXd Mask = MatrixXd::Zero(X.rows(),1);
    MatrixXcd T1r = MatrixXcd::Zero(X.rows(),1);

    // Temp variables
    MatrixXcd Xs = MatrixXcd::Zero(X.rows(),1);
    MatrixXcd Ys = MatrixXcd::Zero(X.rows(),1);
    Xs.array()(indexing::all,0) += std::cos(3.1416/4)*X(indexing::all,0).array() + std::sin(3.1416/4)*X(indexing::all,1).array();
    Ys.array()(indexing::all,0) += -std::sin(3.1416/4)*X(indexing::all,0).array() + std::cos(3.1416/4)*X(indexing::all,1).array();
    // Xs.array()(indexing::all,0) += X(indexing::all,0).array();
    // Ys.array()(indexing::all,0) += X(indexing::all,1).array();    
    MatrixXcd tmp1 = MatrixXcd::Zero(X.rows(),1);
    MatrixXcd tmp2 = MatrixXcd::Zero(X.rows(),1);
    MatrixXcd tmp3 = MatrixXcd::Zero(X.rows(),1);    

    // Complex unit
    std::complex<double> cu(0, 1);
    double sin2b = std::pow(std::sin(beta),2);
    double cos2b = std::pow(std::cos(beta),2);

    // Build magnetizations
    tmp1.array()(indexing::all,0) += M0.array()*sin2b*exp_T1.array()*((ke(0)*Xs(indexing::all,0)).array().cos()*(ke(1)*Ys(indexing::all,0)).array().sin());
    tmp2.array()(indexing::all,0) += M0.array()*(1.0-exp_T1.array()) + M0.array()*cos2b*exp_T1.array();
    tmp3.array()(indexing::all,0) += (cu*T1).array().exp();

    Mxy0 += (tmp1 + tmp2)*prod*std::sin(alpha);
    Mxy1 += (-tmp1 + tmp2)*prod*std::sin(alpha);
    T1r  += tmp3;
    Mask.array() += 1.0;

    return std::make_tuple(Mxy0, Mxy1, Mask, T1r);

}


// Evaluate the DENSE magnetizations on all the spins
Magnetization_Images DENSE_magnetizations(const double &alpha,
                                    const double &prod,
                                    const VectorXd &M0,
                                    const VectorXd &exp_T1,
                                    const VectorXd &ke,
                                    const MatrixXd &X,
                                    const MatrixXd &u,
                                    const MatrixXd &phi){

    // Output images
    MatrixXcd Mxy0 = MatrixXcd::Zero(X.rows(),X.cols());
    MatrixXcd Mxy1 = MatrixXcd::Zero(X.rows(),X.cols());;
    MatrixXd Mask = MatrixXd::Zero(X.rows(),X.cols());;

    // Temp variables
    MatrixXcd tmp1 = MatrixXcd::Zero(X.rows(),X.cols());
    MatrixXcd tmp2 = MatrixXcd::Zero(X.rows(),X.cols());

    // Complex unit
    std::complex<double> cu(0, 1);

    // Build magnetizations
    for (int i=0; i<ke.size(); i++){
        tmp1.array()(indexing::all,i) += 0.5*M0.array()*exp_T1.array()*(-cu*(ke(i)*u(indexing::all,i) + phi)).array().exp();
        tmp1.array()(indexing::all,i) += 0.5*M0.array()*exp_T1.array()*(-cu*(ke(i)*(2*X(indexing::all,i) + u(indexing::all,i)) + phi).array()).exp();
        tmp2.array()(indexing::all,i) += M0.array()*(1-exp_T1.array())*(-cu*(ke(i)*(X(indexing::all,i) + u(indexing::all,i)) + phi).array()).exp();
    }
    Mxy0 += (tmp1 + tmp2)*prod*std::sin(alpha);
    Mxy1 += (-tmp1 + tmp2)*prod*std::sin(alpha);
    Mask.array() += 1;

    return std::make_tuple(Mxy0, Mxy1, Mask);

}


// Exact magnetization
Exact_Images EXACT_magnetizations(const VectorXd &M0,
                                  const VectorXd &ke,
                                  const MatrixXd &u){

    // Output images
    MatrixXcd Mxy = MatrixXcd::Zero(u.rows(),u.cols());
    MatrixXd Mask = MatrixXd::Zero(u.rows(),u.cols());

    // Complex unit
    std::complex<double> cu(0, 1);

    // Build magnetizations
    for (int i=0; i<ke.size(); i++){
        Mxy.array()(indexing::all,i) += M0.array()*(-cu*ke(i)*u(indexing::all,i)).array().exp();
    }
    Mask.array() += 1;

    return std::make_tuple(Mxy,Mask);

}



PYBIND11_MODULE(ImageBuilding, m) {
    m.doc() = "Utilities for spins-based image generation"; // optional module docstring
    m.def("get_images", &get_images, py::return_value_policy::reference,
      "Calculates the pixel intensity based on the pixel-to-spins connectivity");
    m.def("CSPAMM_magnetizations", &CSPAMM_magnetizations, py::return_value_policy::reference);
    m.def("ORI_O_CSPAMM_magnetizations", &ORI_O_CSPAMM_magnetizations, py::return_value_policy::reference);    
    m.def("DENSE_magnetizations", &DENSE_magnetizations, py::return_value_policy::reference);
    m.def("EXACT_magnetizations", &EXACT_magnetizations, py::return_value_policy::reference);
}
