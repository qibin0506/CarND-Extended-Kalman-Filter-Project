#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /**
     TODO:
     * Calculate the RMSE here.
     */
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;
    
    if (estimations.size() != ground_truth.size() || ground_truth.size() == 0) {
        return rmse;
    }
    
    int size = ground_truth.size();
    for (int i = 0; i < size; i++) {
        VectorXd diff = estimations[i] - ground_truth[i];
        diff = diff.array() * diff.array();
        rmse += diff;
    }
    
    rmse = rmse / size;
    rmse = rmse.array().sqrt();
    
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    /**
     TODO:
     * Calculate a Jacobian here.
     */
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    
    MatrixXd Hj(3, 4);
    
    float px2_plus_py2 = px * px + py * py;
    float px2_plus_py2_sqrt = sqrt(px2_plus_py2);
    
    if (fabs(px2_plus_py2) < 0.0001) {
        return Hj;
    }
    
    Hj(0, 0) = px / px2_plus_py2_sqrt;
    Hj(0, 1) = py / px2_plus_py2_sqrt;
    Hj(0, 2) = 0;
    Hj(0, 3) = 0;
    
    Hj(1, 0) = - py / px2_plus_py2;
    Hj(1, 1) = px / px2_plus_py2;
    Hj(1, 2) = 0;
    Hj(1, 3) = 0;
    
    Hj(2, 0) = py * (vx * py - vy * px) / px2_plus_py2_sqrt * px2_plus_py2_sqrt * px2_plus_py2_sqrt;
    Hj(2, 1) = px * (vy * px - vx * py) / px2_plus_py2_sqrt * px2_plus_py2_sqrt * px2_plus_py2_sqrt;
    Hj(2, 2) = px / px2_plus_py2_sqrt;
    Hj(2, 3) = py / px2_plus_py2_sqrt;
    
    return Hj;
}
