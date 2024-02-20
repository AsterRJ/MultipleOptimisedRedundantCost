#ifndef _NULLSPACE_CALCULUS_
#define _NULLSPACE_CALCULUS_

#include <math.h>
#include <eigen3/Eigen/Dense>
#include <Eigen/QR>

using namespace std;
using namespace Eigen;

class NullSpaceCalc{

public:


NullSpaceCalc(){
    th_max << 2.967, 2.094, 2.967, 2.094, 2.967, 2.094,3.054; 
    th_min << -2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054;
}


int plus_(int a, int b){
    // cute test function
    return a+b;
}



void calculate_descent(Eigen::Matrix<double,7,1> q, Eigen::Matrix<double, 3,1> & c_vec, Eigen::Matrix<double,7,1> & desc_vec){
    calc_derivatives(q);
    double c = 0;
    Eigen::Matrix<double,7,1> dc, dc_temp;
    Eigen::Matrix<double,7,7> H, H_temp;
    dc.setZero();
    H.setZero();
    condition_number(q,c, dc_temp, H_temp);
    H+=H_temp;
    dc+=dc_temp;
    c_vec(0) = c;
    manipulability(q,c, dc_temp, H_temp);
    H+=H_temp;
    dc+=dc_temp;
    c_vec(1) = c;
    centreing_coords(q,c, dc_temp, H_temp);
    H+=H_temp;
    dc+=dc_temp;
    c_vec(2) = c;

    // second order optimisation - tested in simulation and it is incomparable in terms of optimisation rate compared with first order.
    desc_vec = - 2*H*dc;
}


void centreing_coords(Eigen::Matrix<double, 7,1>q, double &c, Eigen::Matrix<double,7,1> &dc, Eigen::Matrix<double,7,7> &H){
    // simple cost function that tries to push the direction of motion away from boundaries. I can see some fun with exponentials being more versatile yet unpredictable - will see, not urgent.
    c = 0;
    double c_now, inv_max, inv_min;
    for (int i = 0; i<7; i++)
    {
        inv_max = 1/(th_max(i) - q(i));
        inv_min = 1/(q(i) - th_min(i));

        c_now = (th_max(i) - th_min(i))*(th_max(i) - th_min(i))*inv_max*inv_min;
        c+=c_now;

        dc(i) = c*(inv_max - inv_min);
        H(i,i) = c*(inv_max - inv_min)*(inv_max - inv_min) + c*(inv_max*inv_max + inv_min*inv_min);
    }    
}

void condition_number(Eigen::Matrix<double,7,1> q, double &c, Eigen::Matrix<double,7,1> &dc, Eigen::Matrix<double,7,7> &H)
{
    Eigen::Matrix<double, 7*7, 6> dJ_inv;
    Eigen::Matrix<double, 7*7*7, 6> ddJ_inv;
    Eigen::Matrix<double,7,6> Jinv_L, Jinv_R;
    Jinv_L = PseudoInverse(Jac);
    Jinv_R = (PseudoInverse(Jac.transpose())).transpose();
    inverse_deriviatives(dJ_inv,ddJ_inv,Jinv_L,Jinv_R);
    double jstar,jinvstar;
    jstar = star(Jac,Jac);
    jinvstar = star(Jinv_L, Jinv_L);
    for (int i = 0; i<7; i++){
        double d_a = 2*star(dJ.block<6,7>(6*i,0), Jac)*jinvstar;
        d_a += 2*star(dJ_inv.block<7,6>(7*i,0), Jinv_L)*jstar;
        dc(i) = d_a;
        for (int j = i; j < 7; j++){
            double dd_a = 2*jinvstar*(star(ddJ.block<6,7>(6*(i*7 + j),0), Jac) + star(dJ.block<6,7>(6*i,0), dJ.block<6,7>(6*j,0)));
            dd_a += 2*jstar*(star(ddJ_inv.block<7,6>(7*(7*i + j), 0), Jinv_L) + star(dJ_inv.block<7,6>(7*i,0), dJ_inv.block<7,6>(7*j,0)));
            dd_a += 4*(star(dJ_inv.block<7,6>(7*i,0), Jinv_L)*star(dJ.block<7,6>(7*j,0), Jac) + star(dJ_inv.block<7,6>(7*j,0), Jinv_L)*star(dJ.block<7,6>(7*i,0), Jac));
            H(i,j) = dd_a;
            H(j,i) = dd_a;
        }
    }
}

void manipulability(Eigen::Matrix<double,7,1> q, double &c, Eigen::Matrix<double,7,1> &dc, Eigen::Matrix<double,7,7> &H){
    dc.setZero();
    H.setZero();
    Eigen::Matrix<double, 6, 6> JJT, JJTinv;
    JJT  = Jac*Jac.transpose();
    try{JJTinv = JJT.inverse();throw 1;}
    catch(...){JJTinv = PseudoInverse(JJT);}

    double detJJT = JJT.determinant();
    if(detJJT==0.0){c = DBL_MAX; return;}
    c = 1/(detJJT*detJJT);

    Eigen::Matrix<double,7,1> g_vec = g_i(q, JJTinv);
    Eigen::Matrix<double,7,7> dg_mat = dg(q,JJTinv);

    for (int i = 0;i<7;i++)
    {
        dc(i) = - g_vec(i)/c;
        for (int j = i; j<7; j++)
        {
            H(i,j) = (g_vec(i)*g_vec(j) - dg_mat(i,j))/c;
            H(j,i) = (g_vec(i)*g_vec(j) - dg_mat(i,j))/c;
        }
    }
};



private:

Eigen::Matrix<double,6,7> Jac;
Eigen::Matrix<double,6*7,7> dJ;
Eigen::Matrix<double, 6*(7*7),7> ddJ;
Eigen::Matrix<double, 7,1> th_max, th_min;


MatrixXd PseudoInverse(MatrixXd matrixInput)
{
    MatrixXd matrixInputInv;
    matrixInputInv = matrixInput.completeOrthogonalDecomposition().pseudoInverse();
    return matrixInputInv;
}

double star(MatrixXd A, MatrixXd B){
    int sz_r = A.rows();
    int sz_c = A.cols();
    double ans_ = 0;
    for (int i =0 ; i<sz_r; i++){
        for (int j = 0; j<sz_c; j++)
        {
            ans_+=A(i,j)*B(i,j);
        }
    }
    return ans_;
}


////////////////////////////////////////////////////////////////////////////////////////////
// Condition Number calcs;
////////////////////////////////////////////////////////////////////////////////////////////


void inverse_deriviatives(Eigen::Matrix<double, 7*7, 6> &dJ_inv, Eigen::Matrix<double, 7*7*7, 6> &ddJ_inv, Eigen::Matrix<double, 7,6> J_invL, Eigen::Matrix<double, 7,6> J_invR){
    Eigen::Matrix<double,7,6> ddJ;
    for (int i = 0; i<7; i++){
        dJ_inv.block<7,6>(7*i,0) = - J_invL*dJ.block<6,7>(6*i,0)*J_invR;
        for (int j = i; j<7; j++){
            ddJ = J_invL*dJ.block<6,7>(6*i,0)*J_invR*dJ.block<6,7>(6*j,0)*J_invR + J_invL*dJ.block<6,7>(6*j,0)*J_invR*dJ.block<6,7>(6*i,0)*J_invR - J_invL*ddJ.block<6,7>(6*(7*i + j),0)*J_invR; 
            ddJ_inv.block<7,6>(6*(7*i + j),0) = ddJ;
            ddJ_inv.block<7,6>(6*(7*j + i),0) = ddJ;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////
// Manipulability calcs;
////////////////////////////////////////////////////////////////////////////////////////////

Eigen::Matrix<double,7,1> g_i(Eigen::Matrix<double,7,1> q, Eigen::Matrix<double, 6, 6> JJTinv)
{
    Eigen::Matrix<double,7,1> g_vec;
    Eigen::Matrix<double, 6,6> Mat1, Mat2, Mat3;
    Eigen::Matrix<double,6,7> DerivMat;
    for (int i = 0;i<7; i++)
    {
        DerivMat = dJ.block<6,7>(6*i,0);
        Mat1 = DerivMat*Jac.transpose();
        Mat2 = Jac*DerivMat.transpose();
        Mat3 = JJTinv*(Mat1 + Mat2);
        g_vec(i) = Mat3.trace();
    }
    return g_vec;
}

Eigen::Matrix<double,7,7> dg(Eigen::Matrix<double,7,1> q, Eigen::Matrix<double, 6, 6> JJTinv)
{
    Eigen::Matrix<double,7,7> dg_mat;
    Eigen::Matrix<double, 6,6> Mati1, Matj1, Matj2, Mat3,Mat4, Mat_;
    Eigen::Matrix<double,6,7> DerivMat_i, DerivMat_j, DDerivMat_ij;

    for (int i = 0;i<7; i++)
    {
        
        DerivMat_i = dJ.block<6,7>(6*i,0);
        Mati1 = JJTinv*(DerivMat_i*Jac.transpose() + Jac*DerivMat_i.transpose());
        for (int j = i; j<7; j++)
        {
            DDerivMat_ij = ddJ.block<6,7>(6*(i*7+j),0);
            DerivMat_j = dJ.block<6,7>(6*j,0);
            Matj1 = JJTinv*(DerivMat_j*Jac.transpose() + Jac*DerivMat_j.transpose());
            Mat3 = Matj1*Mati1;


            Mat_ = DDerivMat_ij*Jac.transpose() + DerivMat_i*DerivMat_j.transpose() + DerivMat_j*DerivMat_i.transpose() + Jac*DDerivMat_ij.transpose();
            //Mat_ = DDerivMat_ij*(Jac.transpose() + DerivMat_i*DerivMat_j.transpose() + DerivMat_j*DerivMat_i.transpose() + Jac*DDerivMat_ij.transpose()); 
            Mat4 = JJTinv*Mat_;

            dg_mat(i,j) = Mat4.trace() - Mat3.trace();
            dg_mat(j,i) = Mat4.trace() - Mat3.trace();
        }
    }
    return dg_mat;
}


////////////////////////////////////////////////////////////////////////////////////////////
// Jacobian & derivatives;
////////////////////////////////////////////////////////////////////////////////////////////

void calc_derivatives(Eigen::Matrix<double,7,1> q){
    get_J(q,Jac);
    for (int i = 0; i<7; i++){
        Eigen::Matrix<double,6,7> dJ_i = get_dJ(q,i);
        dJ.block<6,7>(i*7,0) = dJ_i;
        for (int j = i; j<7; j++)
        {
            Eigen::Matrix<double,6,7> ddJ_i_j = get_ddJ(q,i,j);
            ddJ.block<6,7>(6*(7*i+j),0) = ddJ_i_j;
            ddJ.block<6,7>(6*(i+j*7),0) = ddJ_i_j;
        }
    }
}

Eigen::Matrix<double,6,7> get_dJ(Eigen::Matrix<double,7,1> q, int ax)
{
    Eigen::Matrix<double,6,7> dJ;
    switch(ax){
        case 0:
            getJacobian_deriv_0(q,dJ);
            break;
        case 1:
            getJacobian_deriv_1(q,dJ);
            break;
        case 2:
            getJacobian_deriv_2(q,dJ);
            break;
        case 3:
            getJacobian_deriv_3(q,dJ);
            break;
        case 4:
            getJacobian_deriv_4(q,dJ);
            break;
        case 5:
            getJacobian_deriv_5(q,dJ);
            break;
        case 6:
            getJacobian_deriv_6(q,dJ);
            break;
    };
    return dJ;
}

Eigen::Matrix<double,6,7> get_ddJ(Eigen::Matrix<double,7,1> q, int ax1, int ax2)
{
    Eigen::Matrix<double,6,7> ddJ;
    switch(ax1){
        case 0:
            switch(ax2){
                case 0:
                    getJacobian_dblderiv_0_0(q,ddJ);
                    break;
                case 1:
                    getJacobian_dblderiv_0_1(q,ddJ);
                    break;
                case 2:
                    getJacobian_dblderiv_0_2(q,ddJ);
                    break;
                case 3:
                    getJacobian_dblderiv_0_3(q,ddJ);
                    break;
                case 4:
                    getJacobian_dblderiv_0_4(q,ddJ);
                    break;
                case 5:
                    getJacobian_dblderiv_0_5(q,ddJ);
                    break;
                case 6:
                    getJacobian_dblderiv_0_6(q,ddJ);
                    break;
            };
            break;
        case 1:
            switch(ax2){
                case 0:
                    getJacobian_dblderiv_0_1(q,ddJ);
                    break;
                case 1:
                    getJacobian_dblderiv_1_1(q,ddJ);
                    break;
                case 2:
                    getJacobian_dblderiv_1_2(q,ddJ);
                    break;
                case 3:
                    getJacobian_dblderiv_1_3(q,ddJ);
                    break;
                case 4:
                    getJacobian_dblderiv_1_4(q,ddJ);
                    break;
                case 5:
                    getJacobian_dblderiv_1_5(q,ddJ);
                    break;
                case 6:
                    getJacobian_dblderiv_1_6(q,ddJ);
                    break;
            };
            break;
        case 2:
            switch(ax2){
                case 0:
                    getJacobian_dblderiv_0_2(q,ddJ);
                    break;
                case 1:
                    getJacobian_dblderiv_1_2(q,ddJ);
                    break;
                case 2:
                    getJacobian_dblderiv_2_2(q,ddJ);
                    break;
                case 3:
                    getJacobian_dblderiv_2_3(q,ddJ);
                    break;
                case 4:
                    getJacobian_dblderiv_2_4(q,ddJ);
                    break;
                case 5:
                    getJacobian_dblderiv_2_5(q,ddJ);
                    break;
                case 6:
                    getJacobian_dblderiv_2_6(q,ddJ);
                    break;
            };
            break;
        case 3:
            switch(ax2){
                case 0:
                    getJacobian_dblderiv_0_3(q,ddJ);
                    break;
                case 1:
                    getJacobian_dblderiv_1_3(q,ddJ);
                    break;
                case 2:
                    getJacobian_dblderiv_2_3(q,ddJ);
                    break;
                case 3:
                    getJacobian_dblderiv_3_3(q,ddJ);
                    break;
                case 4:
                    getJacobian_dblderiv_3_4(q,ddJ);
                    break;
                case 5:
                    getJacobian_dblderiv_3_5(q,ddJ);
                    break;
                case 6:
                    getJacobian_dblderiv_3_6(q,ddJ);
                    break;
            };
            break;
        case 4:
            switch(ax2){
                case 0:
                    getJacobian_dblderiv_0_4(q,ddJ);
                    break;
                case 1:
                    getJacobian_dblderiv_1_4(q,ddJ);
                    break;
                case 2:
                    getJacobian_dblderiv_2_4(q,ddJ);
                    break;
                case 3:
                    getJacobian_dblderiv_3_4(q,ddJ);
                    break;
                case 4:
                    getJacobian_dblderiv_4_4(q,ddJ);
                    break;
                case 5:
                    getJacobian_dblderiv_4_5(q,ddJ);
                    break;
                case 6:
                    getJacobian_dblderiv_4_6(q,ddJ);
                    break;
            };
            break;
        case 5:
            switch(ax2){
                case 0:
                    getJacobian_dblderiv_0_5(q,ddJ);
                    break;
                case 1:
                    getJacobian_dblderiv_1_5(q,ddJ);
                    break;
                case 2:
                    getJacobian_dblderiv_2_5(q,ddJ);
                    break;
                case 3:
                    getJacobian_dblderiv_3_5(q,ddJ);
                    break;
                case 4:
                    getJacobian_dblderiv_4_5(q,ddJ);
                    break;
                case 5:
                    getJacobian_dblderiv_5_5(q,ddJ);
                    break;
                case 6:
                    getJacobian_dblderiv_5_6(q,ddJ);
                    break;
            };
            break;
        case 6:
            switch(ax2){
                case 0:
                    getJacobian_dblderiv_0_6(q,ddJ);
                    break;
                case 1:
                    getJacobian_dblderiv_1_6(q,ddJ);
                    break;
                case 2:
                    getJacobian_dblderiv_2_6(q,ddJ);
                    break;
                case 3:
                    getJacobian_dblderiv_3_6(q,ddJ);
                    break;
                case 4:
                    getJacobian_dblderiv_4_6(q,ddJ);
                    break;
                case 5:
                    getJacobian_dblderiv_5_6(q,ddJ);
                    break;
                case 6:
                    getJacobian_dblderiv_6_6(q,ddJ);
                    break;
            };
            break;
    };
    return ddJ;
}

void get_J(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
J.setZero();
J(0,0)=0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(0)) - 0.00043624*sin(q(2))*cos(q(0));
J(0,1)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*cos(q(0));
J(0,2)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
J(0,3)=-1.0*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
J(1,0)=-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0));
J(1,1)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0));
J(1,2)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
J(1,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
J(1,4)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
J(2,1)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
J(2,2)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)));
J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)));
J(3,1)=-sin(q(0));
J(3,2)=sin(q(1))*cos(q(0));
J(3,3)=-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1));
J(3,4)=-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3));
J(3,5)=-1.0*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
J(3,6)=(((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) + (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
J(4,1)=cos(q(0));
J(4,2)=sin(q(0))*sin(q(1));
J(4,3)=-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2));
J(4,4)=-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3));
J(4,5)=-1.0*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*cos(q(4));
J(4,6)=(((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) + (-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
J(5,0)=1.0;
J(5,2)=cos(q(1));
J(5,3)=sin(q(1))*sin(q(2));
J(5,4)=sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3));
J(5,5)=-1.0*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*sin(q(4)) + sin(q(1))*sin(q(2))*cos(q(4));
J(5,6)=((-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*cos(q(4)) + sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5)) + (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*cos(q(5));
};


////////////////////////////////////////////////////////////////////////////////////////////
// Jacobian derivatives;
////////////////////////////////////////////////////////////////////////////////////////////


void getJacobian_deriv_0(Eigen::Matrix<double,7,1> q, Eigen::Matrix<double,6,7> &J){ 
J.setZero();
    J(0,0)=0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*cos(q(0));
    J(0,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0));
    J(0,2)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) - (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(0,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(0)) - 0.00043624*sin(q(2))*cos(q(0));
    J(1,1)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*cos(q(0));
    J(1,2)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(1,3)=(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(2,1)=-1.0*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(0)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*cos(q(0));
    J(2,2)=-1.0*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0))*sin(q(1));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3))) + (-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)));
    J(3,1)=-cos(q(0));
    J(3,2)=-sin(q(0))*sin(q(1));
    J(3,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(3,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(3,5)=-1.0*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4));
    J(3,6)=(((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) + (-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
    J(4,1)=-sin(q(0));
    J(4,2)=sin(q(1))*cos(q(0));
    J(4,3)=-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1));
    J(4,4)=-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3));
    J(4,5)=-1.0*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(4,6)=(((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) + (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_0_0(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=-1.0*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*cos(q(0));
    J(1,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(2,1)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1))) + 2*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)));
    J(3,1)=sin(q(0));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
    J(3,5)=-1.0*((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=(((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
    J(4,1)=-cos(q(0));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(4,5)=-1.0*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4));
    J(4,6)=(((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) + (-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_0_1(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=-1.0*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*cos(q(0));
    J(1,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(2,1)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1))) + 2*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)));
    J(3,1)=sin(q(0));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
    J(3,5)=-1.0*((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=(((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
    J(4,1)=-cos(q(0));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(4,5)=-1.0*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4));
    J(4,6)=(((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) + (-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_0_2(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
J.setZero(); 
J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0)) + 0.00043624*sin(q(2))*cos(q(0));
J(0,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*cos(q(0));
J(0,2)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
J(0,3)=-1.0*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
J(0,4)=(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*cos(q(0));
J(1,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0));
J(1,2)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
J(1,3)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
J(1,4)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
J(2,1)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
J(2,2)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1))) + 2*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)));
J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)));
J(3,1)=sin(q(0));
J(3,2)=-sin(q(1))*cos(q(0));
J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
J(3,5)=-1.0*((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
J(3,6)=(((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
J(4,1)=-cos(q(0));
J(4,2)=-sin(q(0))*sin(q(1));
J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
J(4,5)=-1.0*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4));
J(4,6)=(((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) + (-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_0_3(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=-1.0*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*cos(q(0));
    J(1,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(2,1)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1))) + 2*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)));
    J(3,1)=sin(q(0));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
    J(3,5)=-1.0*((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=(((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
    J(4,1)=-cos(q(0));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(4,5)=-1.0*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4));
    J(4,6)=(((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) + (-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_0_4(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=-1.0*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*cos(q(0));
    J(1,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(2,1)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1))) + 2*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)));
    J(3,1)=sin(q(0));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
    J(3,5)=-1.0*((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=(((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
    J(4,1)=-cos(q(0));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(4,5)=-1.0*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4));
    J(4,6)=(((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) + (-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_0_5(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=-1.0*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*cos(q(0));
    J(1,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(2,1)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1))) + 2*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)));
    J(3,1)=sin(q(0));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
    J(3,5)=-1.0*((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=(((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
    J(4,1)=-cos(q(0));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(4,5)=-1.0*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4));
    J(4,6)=(((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) + (-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_0_6(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=-1.0*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*cos(q(0));
    J(1,1)=-1.0*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=(0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(2,1)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1))) + 2*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)));
    J(3,1)=sin(q(0));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
    J(3,5)=-1.0*((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=(((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) + (-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
    J(4,1)=-cos(q(0));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(4,5)=-1.0*((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4));
    J(4,6)=(((-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) + (-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_deriv_1(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=-0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) - 0.4*sin(q(0))*cos(q(1))*cos(q(3)) - 0.42*sin(q(0))*cos(q(1));
    J(0,1)=(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*cos(q(0));
    J(0,2)=(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*cos(q(1)) - (0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*cos(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1));
    J(0,3)=-1.0*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(2))*cos(q(1)) + (-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1))*sin(q(2)) - (0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(1))*sin(q(2));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*cos(q(3)) - sin(q(3))*cos(q(1))*cos(q(2))) + (-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*(sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + sin(q(0))*cos(q(1))*cos(q(3)));
    J(1,0)=0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1));
    J(1,1)=(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*sin(q(0));
    J(1,2)=-1.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*cos(q(0))*cos(q(1)) + (0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*cos(q(1)) - (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1));
    J(1,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(2))*cos(q(1)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2))) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(1))*sin(q(2))*cos(q(0)) + (0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(1))*sin(q(2));
    J(1,4)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*cos(q(3)) + sin(q(3))*cos(q(1))*cos(q(2))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - cos(q(0))*cos(q(1))*cos(q(3))) + (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)));
    J(2,1)=-1.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*sin(q(0)) - (0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*cos(q(0));
    J(2,2)=(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*sin(q(1))*cos(q(0)) - (0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*cos(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*cos(q(0))*cos(q(1));
    J(2,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(1))*sin(q(2)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2))*cos(q(0)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3))) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - sin(q(0))*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + cos(q(0))*cos(q(1))*cos(q(3))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)));
    J(3,2)=cos(q(0))*cos(q(1));
    J(3,3)=sin(q(1))*sin(q(2))*cos(q(0));
    J(3,4)=sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + cos(q(0))*cos(q(1))*cos(q(3));
    J(3,5)=-1.0*(-sin(q(1))*cos(q(0))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1)))*sin(q(4)) + sin(q(1))*sin(q(2))*cos(q(0))*cos(q(4));
    J(3,6)=((-sin(q(1))*cos(q(0))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1)))*cos(q(4)) + sin(q(1))*sin(q(2))*sin(q(4))*cos(q(0)))*sin(q(5)) + (sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + cos(q(0))*cos(q(1))*cos(q(3)))*cos(q(5));
    J(4,2)=sin(q(0))*cos(q(1));
    J(4,3)=sin(q(0))*sin(q(1))*sin(q(2));
    J(4,4)=sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + sin(q(0))*cos(q(1))*cos(q(3));
    J(4,5)=-1.0*(-sin(q(0))*sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1)))*sin(q(4)) + sin(q(0))*sin(q(1))*sin(q(2))*cos(q(4));
    J(4,6)=((-sin(q(0))*sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1)))*cos(q(4)) + sin(q(0))*sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5)) + (sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + sin(q(0))*cos(q(1))*cos(q(3)))*cos(q(5));
    J(5,2)=-sin(q(1));
    J(5,3)=sin(q(2))*cos(q(1));
    J(5,4)=-sin(q(1))*cos(q(3)) + sin(q(3))*cos(q(1))*cos(q(2));
    J(5,5)=-1.0*(-sin(q(1))*sin(q(3)) - cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(2))*cos(q(1))*cos(q(4));
    J(5,6)=((-sin(q(1))*sin(q(3)) - cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(2))*sin(q(4))*cos(q(1)))*sin(q(5)) + (-sin(q(1))*cos(q(3)) + sin(q(3))*cos(q(1))*cos(q(2)))*cos(q(5));
};


void getJacobian_dblderiv_1_1(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) - 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*sin(q(0))*cos(q(1)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) + (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*sin(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1))*sin(q(2)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) - 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + sin(q(0))*cos(q(1))*cos(q(3))) + 2*(sin(q(1))*cos(q(3)) - sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3))) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=-2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*cos(q(0))*cos(q(1)) - (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(1)) - (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) - 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*cos(q(0)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(1,4)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + 2*(-sin(q(1))*cos(q(3)) + sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - cos(q(0))*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*(sin(q(1))*cos(q(0))*cos(q(3)) - sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)));
    J(2,1)=-1.0*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0));
    J(2,2)=(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*cos(q(0))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(0))*cos(q(1)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(1))*sin(q(2))*cos(q(0)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1))*sin(q(2));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(1))*cos(q(3)) - sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2*(-sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - sin(q(0))*cos(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + cos(q(0))*cos(q(1))*cos(q(3)));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2));
    J(3,5)=-1.0*(-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(2))*cos(q(0))*cos(q(1))*cos(q(4));
    J(3,6)=((-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(2))*sin(q(4))*cos(q(0))*cos(q(1)))*sin(q(5)) + (-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1));
    J(4,4)=-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2));
    J(4,5)=-1.0*(-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(0))*sin(q(2))*cos(q(1))*cos(q(4));
    J(4,6)=((-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(0))*sin(q(2))*sin(q(4))*cos(q(1)))*sin(q(5)) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(5,2)=-cos(q(1));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3));
    J(5,5)=-1.0*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(4)) - sin(q(1))*sin(q(2))*cos(q(4));
    J(5,6)=((sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*cos(q(4)) - sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5)) + (-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_1_2(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) - 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*sin(q(0))*cos(q(1)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) + (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*sin(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1))*sin(q(2)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) - 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + sin(q(0))*cos(q(1))*cos(q(3))) + 2*(sin(q(1))*cos(q(3)) - sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3))) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=-2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*cos(q(0))*cos(q(1)) - (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(1)) - (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) - 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*cos(q(0)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(1,4)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + 2*(-sin(q(1))*cos(q(3)) + sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - cos(q(0))*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*(sin(q(1))*cos(q(0))*cos(q(3)) - sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)));
    J(2,1)=-1.0*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0));
    J(2,2)=(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*cos(q(0))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(0))*cos(q(1)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(1))*sin(q(2))*cos(q(0)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1))*sin(q(2));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(1))*cos(q(3)) - sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2*(-sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - sin(q(0))*cos(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + cos(q(0))*cos(q(1))*cos(q(3)));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2));
    J(3,5)=-1.0*(-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(2))*cos(q(0))*cos(q(1))*cos(q(4));
    J(3,6)=((-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(2))*sin(q(4))*cos(q(0))*cos(q(1)))*sin(q(5)) + (-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1));
    J(4,4)=-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2));
    J(4,5)=-1.0*(-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(0))*sin(q(2))*cos(q(1))*cos(q(4));
    J(4,6)=((-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(0))*sin(q(2))*sin(q(4))*cos(q(1)))*sin(q(5)) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(5,2)=-cos(q(1));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3));
    J(5,5)=-1.0*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(4)) - sin(q(1))*sin(q(2))*cos(q(4));
    J(5,6)=((sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*cos(q(4)) - sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5)) + (-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_1_3(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero();
    J(0,0)=0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) - 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*sin(q(0))*cos(q(1)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) + (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*sin(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1))*sin(q(2)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) - 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + sin(q(0))*cos(q(1))*cos(q(3))) + 2*(sin(q(1))*cos(q(3)) - sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3))) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=-2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*cos(q(0))*cos(q(1)) - (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(1)) - (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) - 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*cos(q(0)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(1,4)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + 2*(-sin(q(1))*cos(q(3)) + sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - cos(q(0))*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*(sin(q(1))*cos(q(0))*cos(q(3)) - sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)));
    J(2,1)=-1.0*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0));
    J(2,2)=(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*cos(q(0))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(0))*cos(q(1)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(1))*sin(q(2))*cos(q(0)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1))*sin(q(2));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(1))*cos(q(3)) - sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2*(-sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - sin(q(0))*cos(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + cos(q(0))*cos(q(1))*cos(q(3)));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2));
    J(3,5)=-1.0*(-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(2))*cos(q(0))*cos(q(1))*cos(q(4));
    J(3,6)=((-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(2))*sin(q(4))*cos(q(0))*cos(q(1)))*sin(q(5)) + (-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1));
    J(4,4)=-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2));
    J(4,5)=-1.0*(-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(0))*sin(q(2))*cos(q(1))*cos(q(4));
    J(4,6)=((-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(0))*sin(q(2))*sin(q(4))*cos(q(1)))*sin(q(5)) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(5,2)=-cos(q(1));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3));
    J(5,5)=-1.0*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(4)) - sin(q(1))*sin(q(2))*cos(q(4));
    J(5,6)=((sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*cos(q(4)) - sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5)) + (-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_1_4(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) - 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*sin(q(0))*cos(q(1)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) + (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*sin(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1))*sin(q(2)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) - 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + sin(q(0))*cos(q(1))*cos(q(3))) + 2*(sin(q(1))*cos(q(3)) - sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3))) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=-2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*cos(q(0))*cos(q(1)) - (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(1)) - (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) - 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*cos(q(0)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(1,4)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + 2*(-sin(q(1))*cos(q(3)) + sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - cos(q(0))*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*(sin(q(1))*cos(q(0))*cos(q(3)) - sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)));
    J(2,1)=-1.0*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0));
    J(2,2)=(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*cos(q(0))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(0))*cos(q(1)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(1))*sin(q(2))*cos(q(0)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1))*sin(q(2));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(1))*cos(q(3)) - sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2*(-sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - sin(q(0))*cos(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + cos(q(0))*cos(q(1))*cos(q(3)));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2));
    J(3,5)=-1.0*(-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(2))*cos(q(0))*cos(q(1))*cos(q(4));
    J(3,6)=((-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(2))*sin(q(4))*cos(q(0))*cos(q(1)))*sin(q(5)) + (-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1));
    J(4,4)=-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2));
    J(4,5)=-1.0*(-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(0))*sin(q(2))*cos(q(1))*cos(q(4));
    J(4,6)=((-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(0))*sin(q(2))*sin(q(4))*cos(q(1)))*sin(q(5)) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(5,2)=-cos(q(1));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3));
    J(5,5)=-1.0*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(4)) - sin(q(1))*sin(q(2))*cos(q(4));
    J(5,6)=((sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*cos(q(4)) - sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5)) + (-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_1_5(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) - 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*sin(q(0))*cos(q(1)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) + (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*sin(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1))*sin(q(2)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) - 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + sin(q(0))*cos(q(1))*cos(q(3))) + 2*(sin(q(1))*cos(q(3)) - sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3))) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=-2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*cos(q(0))*cos(q(1)) - (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(1)) - (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) - 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*cos(q(0)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(1,4)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + 2*(-sin(q(1))*cos(q(3)) + sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - cos(q(0))*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*(sin(q(1))*cos(q(0))*cos(q(3)) - sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)));
    J(2,1)=-1.0*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0));
    J(2,2)=(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*cos(q(0))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(0))*cos(q(1)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(1))*sin(q(2))*cos(q(0)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1))*sin(q(2));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(1))*cos(q(3)) - sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2*(-sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - sin(q(0))*cos(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + cos(q(0))*cos(q(1))*cos(q(3)));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2));
    J(3,5)=-1.0*(-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(2))*cos(q(0))*cos(q(1))*cos(q(4));
    J(3,6)=((-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(2))*sin(q(4))*cos(q(0))*cos(q(1)))*sin(q(5)) + (-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1));
    J(4,4)=-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2));
    J(4,5)=-1.0*(-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(0))*sin(q(2))*cos(q(1))*cos(q(4));
    J(4,6)=((-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(0))*sin(q(2))*sin(q(4))*cos(q(1)))*sin(q(5)) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(5,2)=-cos(q(1));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3));
    J(5,5)=-1.0*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(4)) - sin(q(1))*sin(q(2))*cos(q(4));
    J(5,6)=((sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*cos(q(4)) - sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5)) + (-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_1_6(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) - 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*cos(q(0));
    J(0,2)=2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*sin(q(0))*cos(q(1)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) + (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*sin(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1))*sin(q(2)) - (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) - 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + sin(q(0))*cos(q(1))*cos(q(3))) + 2*(sin(q(1))*cos(q(3)) - sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3))) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(0));
    J(1,2)=-2.0*(-0.4*sin(q(1))*cos(q(3)) - 0.42*sin(q(1)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(1))*cos(q(2)))*cos(q(0))*cos(q(1)) - (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)) - 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)) + 0.42*cos(q(1)))*sin(q(1))*cos(q(0)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(1)) - (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) - 2.0*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*cos(q(0)) - (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2)) + 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(2))*cos(q(1));
    J(1,4)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + 2*(-sin(q(1))*cos(q(3)) + sin(q(3))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(-0.4*sin(q(1))*cos(q(3)) + 0.4*sin(q(3))*cos(q(1))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - cos(q(0))*cos(q(1))*cos(q(3))) + (0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*(sin(q(1))*cos(q(0))*cos(q(3)) - sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)));
    J(2,1)=-1.0*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0));
    J(2,2)=(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) - 0.42*sin(q(0))*sin(q(1)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) - (-0.4*sin(q(1))*cos(q(0))*cos(q(3)) - 0.42*sin(q(1))*cos(q(0)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.00043624*sin(q(0))*sin(q(1))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)) + 0.42*sin(q(0))*cos(q(1)))*cos(q(0))*cos(q(1)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) - 0.00043624*sin(q(1))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)) + 0.42*cos(q(0))*cos(q(1)))*sin(q(0))*cos(q(1)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.42*sin(q(1))*cos(q(0)) + 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.42*sin(q(0))*sin(q(1)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(2))*cos(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(2))*cos(q(0))*cos(q(1)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2.0*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*sin(q(1))*sin(q(2))*cos(q(0)) - 2.0*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1))*sin(q(2));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(0))*sin(q(1))*cos(q(3)) + 0.4*sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(1))*cos(q(3)) - sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*cos(q(0))*cos(q(3)) + 0.4*sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2))) + 2*(-sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) - sin(q(0))*cos(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + 0.4*cos(q(0))*cos(q(1))*cos(q(3))) + 2*(0.4*sin(q(0))*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*sin(q(0))*cos(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(0))*cos(q(2)) + cos(q(0))*cos(q(1))*cos(q(3)));
    J(3,2)=-sin(q(1))*cos(q(0));
    J(3,3)=sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2));
    J(3,5)=-1.0*(-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(2))*cos(q(0))*cos(q(1))*cos(q(4));
    J(3,6)=((-sin(q(1))*sin(q(3))*cos(q(0)) - cos(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(2))*sin(q(4))*cos(q(0))*cos(q(1)))*sin(q(5)) + (-sin(q(1))*cos(q(0))*cos(q(3)) + sin(q(3))*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(4,2)=-sin(q(0))*sin(q(1));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1));
    J(4,4)=-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2));
    J(4,5)=-1.0*(-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*sin(q(4)) + sin(q(0))*sin(q(2))*cos(q(1))*cos(q(4));
    J(4,6)=((-sin(q(0))*sin(q(1))*sin(q(3)) - sin(q(0))*cos(q(1))*cos(q(2))*cos(q(3)))*cos(q(4)) + sin(q(0))*sin(q(2))*sin(q(4))*cos(q(1)))*sin(q(5)) + (-sin(q(0))*sin(q(1))*cos(q(3)) + sin(q(0))*sin(q(3))*cos(q(1))*cos(q(2)))*cos(q(5));
    J(5,2)=-cos(q(1));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3));
    J(5,5)=-1.0*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(4)) - sin(q(1))*sin(q(2))*cos(q(4));
    J(5,6)=((sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*cos(q(4)) - sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5)) + (-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_deriv_2(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2))*cos(q(1)) - 0.00043624*cos(q(0))*cos(q(2));
    J(0,1)=(-0.4*sin(q(1))*sin(q(2))*sin(q(3)) + 0.00043624*sin(q(1))*sin(q(2)))*cos(q(0));
    J(0,2)=(-0.4*sin(q(1))*sin(q(2))*sin(q(3)) + 0.00043624*sin(q(1))*sin(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2))*cos(q(1)) + 0.00043624*cos(q(0))*cos(q(2)))*cos(q(1));
    J(0,3)=-1.0*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*cos(q(2)) + (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=-0.4*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2))*sin(q(3)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2))*sin(q(3)) - 0.4*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*sin(q(3)) - (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3));
    J(1,0)=-0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0))*cos(q(1));
    J(1,1)=(-0.4*sin(q(1))*sin(q(2))*sin(q(3)) + 0.00043624*sin(q(1))*sin(q(2)))*sin(q(0));
    J(1,2)=-1.0*(-0.4*sin(q(1))*sin(q(2))*sin(q(3)) + 0.00043624*sin(q(1))*sin(q(2)))*sin(q(1))*cos(q(0)) + (-0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(1));
    J(1,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*cos(q(2)) + (-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(2))*sin(q(3)) - 0.4*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(2))*sin(q(3));
    J(1,4)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2))*sin(q(3)) - 0.4*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2))*sin(q(3)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3)) - 0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*sin(q(3));
    J(2,1)=-1.0*(-0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(0)) - (-0.4*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2))*cos(q(1)) + 0.00043624*cos(q(0))*cos(q(2)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(0))*sin(q(1)) + (-0.4*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) - 0.00043624*sin(q(0))*sin(q(2))*cos(q(1)) + 0.00043624*cos(q(0))*cos(q(2)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2))) - 0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(3));
    J(2,4)=-0.4*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(3)) - 0.4*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(3));
    J(3,3)=sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2));
    J(3,4)=-1.0*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(3));
    J(3,5)=(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(4)) - (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4))*cos(q(3));
    J(3,6)=((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(4)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(3))*cos(q(4)))*sin(q(5)) - (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(3))*cos(q(5));
    J(4,3)=-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0));
    J(4,4)=-1.0*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3));
    J(4,5)=-1.0*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(4))*cos(q(3)) + (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(4));
    J(4,6)=((-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*cos(q(3))*cos(q(4)) + (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(4)))*sin(q(5)) - (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3))*cos(q(5));
    J(5,3)=sin(q(1))*cos(q(2));
    J(5,4)=-sin(q(1))*sin(q(2))*sin(q(3));
    J(5,5)=-sin(q(1))*sin(q(2))*sin(q(4))*cos(q(3)) + sin(q(1))*cos(q(2))*cos(q(4));
    J(5,6)=(sin(q(1))*sin(q(2))*cos(q(3))*cos(q(4)) + sin(q(1))*sin(q(4))*cos(q(2)))*sin(q(5)) - sin(q(1))*sin(q(2))*sin(q(3))*cos(q(5));
};


void getJacobian_dblderiv_2_2(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero();
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*cos(q(0));
    J(0,2)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 0.4*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(1))*sin(q(3))*cos(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(1))*sin(q(2))*sin(q(3));
    J(0,4)=-0.4*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*sin(q(3)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(0));
    J(1,2)=-1.0*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) - 0.8*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*sin(q(3)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.8*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(3))*cos(q(2));
    J(1,4)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) + (sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*sin(q(3));
    J(2,1)=-1.0*(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) - 0.8*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.8*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3));
    J(2,4)=-0.4*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3));
    J(3,5)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(4))*cos(q(3)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3))*cos(q(5)) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3));
    J(4,5)=(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(4))*cos(q(3));
    J(4,6)=((sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)) + (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3))*cos(q(4)))*sin(q(5)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3))*cos(q(5));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2));
    J(5,5)=-sin(q(1))*sin(q(2))*cos(q(4)) - sin(q(1))*sin(q(4))*cos(q(2))*cos(q(3));
    J(5,6)=(-sin(q(1))*sin(q(2))*sin(q(4)) + sin(q(1))*cos(q(2))*cos(q(3))*cos(q(4)))*sin(q(5)) - sin(q(1))*sin(q(3))*cos(q(2))*cos(q(5));
};


void getJacobian_dblderiv_2_3(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*cos(q(0));
    J(0,2)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 0.4*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(1))*sin(q(3))*cos(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(1))*sin(q(2))*sin(q(3));
    J(0,4)=-0.4*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*sin(q(3)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(0));
    J(1,2)=-1.0*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) - 0.8*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*sin(q(3)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.8*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(3))*cos(q(2));
    J(1,4)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) + (sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*sin(q(3));
    J(2,1)=-1.0*(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) - 0.8*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.8*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3));
    J(2,4)=-0.4*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3));
    J(3,5)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(4))*cos(q(3)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3))*cos(q(5)) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3));
    J(4,5)=(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(4))*cos(q(3));
    J(4,6)=((sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)) + (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3))*cos(q(4)))*sin(q(5)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3))*cos(q(5));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2));
    J(5,5)=-sin(q(1))*sin(q(2))*cos(q(4)) - sin(q(1))*sin(q(4))*cos(q(2))*cos(q(3));
    J(5,6)=(-sin(q(1))*sin(q(2))*sin(q(4)) + sin(q(1))*cos(q(2))*cos(q(3))*cos(q(4)))*sin(q(5)) - sin(q(1))*sin(q(3))*cos(q(2))*cos(q(5));
};


void getJacobian_dblderiv_2_4(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*cos(q(0));
    J(0,2)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 0.4*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(1))*sin(q(3))*cos(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(1))*sin(q(2))*sin(q(3));
    J(0,4)=-0.4*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*sin(q(3)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(0));
    J(1,2)=-1.0*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) - 0.8*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*sin(q(3)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.8*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(3))*cos(q(2));
    J(1,4)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) + (sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*sin(q(3));
    J(2,1)=-1.0*(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) - 0.8*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.8*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3));
    J(2,4)=-0.4*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3));
    J(3,5)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(4))*cos(q(3)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3))*cos(q(5)) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3));
    J(4,5)=(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(4))*cos(q(3));
    J(4,6)=((sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)) + (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3))*cos(q(4)))*sin(q(5)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3))*cos(q(5));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2));
    J(5,5)=-sin(q(1))*sin(q(2))*cos(q(4)) - sin(q(1))*sin(q(4))*cos(q(2))*cos(q(3));
    J(5,6)=(-sin(q(1))*sin(q(2))*sin(q(4)) + sin(q(1))*cos(q(2))*cos(q(3))*cos(q(4)))*sin(q(5)) - sin(q(1))*sin(q(3))*cos(q(2))*cos(q(5));
};


void getJacobian_dblderiv_2_5(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*cos(q(0));
    J(0,2)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 0.4*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(1))*sin(q(3))*cos(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(1))*sin(q(2))*sin(q(3));
    J(0,4)=-0.4*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*sin(q(3)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(0));
    J(1,2)=-1.0*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) - 0.8*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*sin(q(3)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.8*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(3))*cos(q(2));
    J(1,4)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) + (sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*sin(q(3));
    J(2,1)=-1.0*(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) - 0.8*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.8*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3));
    J(2,4)=-0.4*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3));
    J(3,5)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(4))*cos(q(3)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3))*cos(q(5)) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3));
    J(4,5)=(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(4))*cos(q(3));
    J(4,6)=((sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)) + (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3))*cos(q(4)))*sin(q(5)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3))*cos(q(5));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2));
    J(5,5)=-sin(q(1))*sin(q(2))*cos(q(4)) - sin(q(1))*sin(q(4))*cos(q(2))*cos(q(3));
    J(5,6)=(-sin(q(1))*sin(q(2))*sin(q(4)) + sin(q(1))*cos(q(2))*cos(q(3))*cos(q(4)))*sin(q(5)) - sin(q(1))*sin(q(3))*cos(q(2))*cos(q(5));
};


void getJacobian_dblderiv_2_6(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero();
    J(0,0)=0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) + 0.00043624*sin(q(2))*cos(q(0));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*cos(q(0));
    J(0,2)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*cos(q(1));
    J(0,3)=(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + 0.4*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(1))*sin(q(3))*cos(q(2)) + (sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(1))*sin(q(2))*sin(q(3));
    J(0,4)=-0.4*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*sin(q(3)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3));
    J(1,0)=-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(0));
    J(1,2)=-1.0*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.00043624*sin(q(1))*cos(q(2)))*sin(q(1))*cos(q(0)) + (-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(1));
    J(1,3)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) - 0.8*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(1))*sin(q(2))*sin(q(3)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) - 0.8*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(1))*sin(q(3))*cos(q(2));
    J(1,4)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(3))*cos(q(2)) + (sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*sin(q(3));
    J(2,1)=-1.0*(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(0)) - (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.00043624*sin(q(0))*sin(q(2)) - 0.00043624*cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(0))*sin(q(1)) + (-0.4*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.00043624*sin(q(0))*cos(q(1))*cos(q(2)) - 0.00043624*sin(q(2))*cos(q(0)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1))) - 0.8*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(3)) - 0.4*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - 0.8*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3));
    J(2,4)=-0.4*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3));
    J(3,3)=sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1));
    J(3,4)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3));
    J(3,5)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(4))*cos(q(3)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=-1.0*(sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3))*cos(q(5)) + ((sin(q(0))*sin(q(2)) - cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3))*cos(q(4)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5));
    J(4,3)=sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2));
    J(4,4)=-1.0*(-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3));
    J(4,5)=(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*cos(q(4)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(4))*cos(q(3));
    J(4,6)=((sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2)))*sin(q(4)) + (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*cos(q(3))*cos(q(4)))*sin(q(5)) - (-sin(q(0))*cos(q(1))*cos(q(2)) - sin(q(2))*cos(q(0)))*sin(q(3))*cos(q(5));
    J(5,3)=-sin(q(1))*sin(q(2));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2));
    J(5,5)=-sin(q(1))*sin(q(2))*cos(q(4)) - sin(q(1))*sin(q(4))*cos(q(2))*cos(q(3));
    J(5,6)=(-sin(q(1))*sin(q(2))*sin(q(4)) + sin(q(1))*cos(q(2))*cos(q(3))*cos(q(4)))*sin(q(5)) - sin(q(1))*sin(q(3))*cos(q(2))*cos(q(5));
};


void getJacobian_deriv_3(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero(); 
    J(0,0)=0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + 0.4*sin(q(0))*sin(q(1))*sin(q(3));
    J(0,1)=(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1)))*cos(q(0));
    J(0,2)=-1.0*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(1)) + (0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1)))*sin(q(0))*sin(q(1));
    J(0,3)=-1.0*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1)));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1))) + (-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)));
    J(1,0)=-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0));
    J(1,1)=(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1)))*sin(q(0));
    J(1,2)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(1)) - (0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1)))*sin(q(1))*cos(q(0));
    J(1,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1)));
    J(1,4)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3)));
    J(2,1)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(0)) - (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(0));
    J(2,2)=-1.0*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(0))*sin(q(1)) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(1))*cos(q(0));
    J(2,3)=(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3))) + (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)));
    J(3,4)=-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0));
    J(3,5)=-1.0*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(4));
    J(3,6)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(5))*cos(q(4)) + (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(5));
    J(4,4)=-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3));
    J(4,5)=-1.0*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(4));
    J(4,6)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(5))*cos(q(4)) + (-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(5));
    J(5,4)=sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1));
    J(5,5)=-1.0*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*sin(q(4));
    J(5,6)=(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*sin(q(5))*cos(q(4)) + (sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*cos(q(5));
};


void getJacobian_dblderiv_3_3(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero();
    J(0,0)=-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*cos(q(0));
    J(0,2)=-1.0*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(1)) + (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1));
    J(0,3)=-1.0*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + 2*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1))) + 2*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3)))*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)));
    J(1,0)=0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(0));
    J(1,2)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(1)) - (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(1))*cos(q(0));
    J(1,3)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + (0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1)));
    J(2,1)=-1.0*(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(0)) - (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(0));
    J(2,2)=-1.0*(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(1)) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*cos(q(0));
    J(2,3)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + (0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)));
    J(3,4)=(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
    J(3,5)=-1.0*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4));
    J(3,6)=((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5)) + (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(5))*cos(q(4));
    J(4,4)=(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(4,5)=-1.0*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4));
    J(4,6)=((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5)) + (-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(5))*cos(q(4));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3));
    J(5,5)=-1.0*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(4));
    J(5,6)=(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*cos(q(5)) + (sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(5))*cos(q(4));
}


void getJacobian_dblderiv_3_4(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero();
    J(0,0)=-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*cos(q(0));
    J(0,2)=-1.0*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(1)) + (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1));
    J(0,3)=-1.0*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + 2*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1))) + 2*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3)))*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)));
    J(1,0)=0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(0));
    J(1,2)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(1)) - (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(1))*cos(q(0));
    J(1,3)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + (0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1)));
    J(2,1)=-1.0*(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(0)) - (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(0));
    J(2,2)=-1.0*(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(1)) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*cos(q(0));
    J(2,3)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + (0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)));
    J(3,4)=(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
    J(3,5)=-1.0*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4));
    J(3,6)=((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5)) + (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(5))*cos(q(4));
    J(4,4)=(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(4,5)=-1.0*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4));
    J(4,6)=((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5)) + (-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(5))*cos(q(4));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3));
    J(5,5)=-1.0*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(4));
    J(5,6)=(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*cos(q(5)) + (sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(5))*cos(q(4));
};


void getJacobian_dblderiv_3_5(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
 { 
    J.setZero(); 
    J(0,0)=-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*cos(q(0));
    J(0,2)=-1.0*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(1)) + (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1));
    J(0,3)=-1.0*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + 2*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1))) + 2*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3)))*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)));
    J(1,0)=0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(0));
    J(1,2)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(1)) - (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(1))*cos(q(0));
    J(1,3)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + (0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1)));
    J(2,1)=-1.0*(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(0)) - (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(0));
    J(2,2)=-1.0*(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(1)) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*cos(q(0));
    J(2,3)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + (0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)));
    J(3,4)=(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
    J(3,5)=-1.0*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4));
    J(3,6)=((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5)) + (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(5))*cos(q(4));
    J(4,4)=(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(4,5)=-1.0*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4));
    J(4,6)=((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5)) + (-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(5))*cos(q(4));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3));
    J(5,5)=-1.0*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(4));
    J(5,6)=(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*cos(q(5)) + (sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(5))*cos(q(4));
};


void getJacobian_dblderiv_3_6(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero();
    J(0,0)=-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3));
    J(0,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*cos(q(0));
    J(0,2)=-1.0*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(1)) + (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(0))*sin(q(1));
    J(0,3)=-1.0*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*sin(q(2)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)));
    J(0,4)=(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + (-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + ((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + 2*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1))) + 2*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3)))*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)));
    J(1,0)=0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3));
    J(1,1)=(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(0));
    J(1,2)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(1)) - (-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)))*sin(q(1))*cos(q(0));
    J(1,3)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(1))*sin(q(2)) + (sin(q(0))*cos(q(2)) + sin(q(2))*cos(q(0))*cos(q(1)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3)));
    J(1,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*sin(q(1))*sin(q(3))*cos(q(2)) + 0.4*cos(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3))) + (0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*sin(q(1))*sin(q(3))*cos(q(2)) - 0.4*cos(q(1))*cos(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1))) + 2*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*(0.4*sin(q(1))*cos(q(2))*cos(q(3)) - 0.4*sin(q(3))*cos(q(1)));
    J(2,1)=-1.0*(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(0)) - (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(0));
    J(2,2)=-1.0*(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(0))*sin(q(1)) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(1))*cos(q(0));
    J(2,3)=(0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(sin(q(0))*sin(q(2))*cos(q(1)) - cos(q(0))*cos(q(2))) + (0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3)))*(-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)));
    J(2,4)=(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*(0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + (-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3))) + (0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - 0.4*sin(q(1))*cos(q(0))*cos(q(3)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3))) + ((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + 0.4*sin(q(0))*sin(q(1))*cos(q(3))) + 2*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*(-0.4*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - 0.4*sin(q(0))*sin(q(1))*sin(q(3))) + 2*(-0.4*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - 0.4*sin(q(1))*sin(q(3))*cos(q(0)))*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)));
    J(3,4)=(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3));
    J(3,5)=-1.0*(-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4));
    J(3,6)=((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) - sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5)) + (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) - sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(5))*cos(q(4));
    J(4,4)=(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3));
    J(4,5)=-1.0*(-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4));
    J(4,6)=((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) - sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5)) + (-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) - sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(5))*cos(q(4));
    J(5,4)=-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3));
    J(5,5)=-1.0*(sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(4));
    J(5,6)=(-sin(q(1))*sin(q(3))*cos(q(2)) - cos(q(1))*cos(q(3)))*cos(q(5)) + (sin(q(1))*cos(q(2))*cos(q(3)) - sin(q(3))*cos(q(1)))*sin(q(5))*cos(q(4));
};


void getJacobian_deriv_4(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero();
    J(3,5)=-1.0*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) - (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4));
    J(3,6)=(-1.0*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4)))*sin(q(5));
    J(4,5)=-1.0*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) - (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(4));
    J(4,6)=(-1.0*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*cos(q(4)))*sin(q(5));
    J(5,5)=-1.0*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*cos(q(4)) - sin(q(1))*sin(q(2))*sin(q(4));
    J(5,6)=(-1.0*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*sin(q(4)) + sin(q(1))*sin(q(2))*cos(q(4)))*sin(q(5));
};


void getJacobian_dblderiv_4_4(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero();
    J(3,5)=((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) - (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=(-1.0*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) - (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5));
    J(4,5)=((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) - (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*cos(q(4));
    J(4,6)=(-1.0*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) - (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5));
    J(5,5)=(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*sin(q(4)) - sin(q(1))*sin(q(2))*cos(q(4));
    J(5,6)=(-1.0*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*cos(q(4)) - sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5));
};


void getJacobian_dblderiv_4_5(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
 { 
    J.setZero();
    J(3,5)=((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) - (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=(-1.0*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) - (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5));
    J(4,5)=((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) - (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*cos(q(4));
    J(4,6)=(-1.0*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) - (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5));
    J(5,5)=(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*sin(q(4)) - sin(q(1))*sin(q(2))*cos(q(4));
    J(5,6)=(-1.0*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*cos(q(4)) - sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5));
};


void getJacobian_dblderiv_4_6(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero();
    J(3,5)=((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*sin(q(4)) - (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*cos(q(4));
    J(3,6)=(-1.0*((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) - (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5));
    J(4,5)=((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*sin(q(4)) - (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*cos(q(4));
    J(4,6)=(-1.0*((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) - (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5));
    J(5,5)=(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*sin(q(4)) - sin(q(1))*sin(q(2))*cos(q(4));
    J(5,6)=(-1.0*(-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*cos(q(4)) - sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5));
};


void getJacobian_deriv_5(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
 { 
    J.setZero();
    J(3,6)=(((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*cos(q(5)) - (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*sin(q(5));
    J(4,6)=(((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(4)))*cos(q(5)) - (-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*sin(q(5));
    J(5,6)=((-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*cos(q(4)) + sin(q(1))*sin(q(2))*sin(q(4)))*cos(q(5)) - (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*sin(q(5));
};


void getJacobian_dblderiv_5_5(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
 { 
    J.setZero();
    J(3,6)=-1.0*(((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) - (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
    J(4,6)=-1.0*(((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) - (-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
    J(5,6)=-1.0*((-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*cos(q(4)) + sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5)) - (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_dblderiv_5_6(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
{ 
    J.setZero();
    J(3,6)=-1.0*(((-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*cos(q(3)) + sin(q(1))*sin(q(3))*cos(q(0)))*cos(q(4)) + (-sin(q(0))*cos(q(2)) - sin(q(2))*cos(q(0))*cos(q(1)))*sin(q(4)))*sin(q(5)) - (-1.0*(-sin(q(0))*sin(q(2)) + cos(q(0))*cos(q(1))*cos(q(2)))*sin(q(3)) + sin(q(1))*cos(q(0))*cos(q(3)))*cos(q(5));
    J(4,6)=-1.0*(((sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*cos(q(3)) + sin(q(0))*sin(q(1))*sin(q(3)))*cos(q(4)) + (-sin(q(0))*sin(q(2))*cos(q(1)) + cos(q(0))*cos(q(2)))*sin(q(4)))*sin(q(5)) - (-1.0*(sin(q(0))*cos(q(1))*cos(q(2)) + sin(q(2))*cos(q(0)))*sin(q(3)) + sin(q(0))*sin(q(1))*cos(q(3)))*cos(q(5));
    J(5,6)=-1.0*((-sin(q(1))*cos(q(2))*cos(q(3)) + sin(q(3))*cos(q(1)))*cos(q(4)) + sin(q(1))*sin(q(2))*sin(q(4)))*sin(q(5)) - (sin(q(1))*sin(q(3))*cos(q(2)) + cos(q(1))*cos(q(3)))*cos(q(5));
};


void getJacobian_deriv_6(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
 { 
    J.setZero();
};


void getJacobian_dblderiv_6_6(Eigen::Matrix<double,7,1> q,Eigen::Matrix<double,6,7> &J)
 { 
    J.setZero();
};


};




#endif