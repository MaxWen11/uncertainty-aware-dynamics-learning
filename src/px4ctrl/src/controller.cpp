#include "controller.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// ==========================================
// OnlineGP Implementation
// ==========================================
OnlineGP::OnlineGP(double l, double sigma_f, double beta, int N_max) 
    : lengthscale(l), var_f(sigma_f), beta_factor(beta), max_window_size(N_max) {
    noise_var = 0.01; 
}

double OnlineGP::kernel_se(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2) {
    return var_f * exp(- (x1 - x2).squaredNorm() / (2.0 * lengthscale * lengthscale));
}

void OnlineGP::add_data(const Eigen::Vector2d& x, double u, double y, double f0, double g0) {
    X_buffer.push_back(x);
    U_buffer.push_back(u);
    Y_buffer.push_back(y);
    F0_buffer.push_back(f0);
    G0_buffer.push_back(g0);
    if (X_buffer.size() > (size_t)max_window_size) {
        X_buffer.pop_front();
        U_buffer.pop_front();
        Y_buffer.pop_front();
        F0_buffer.pop_front();
        G0_buffer.pop_front();
    }
}

void OnlineGP::predict(const Eigen::Vector2d& x_query, double u_query, 
                       double f_prior, double g_prior,
                       double& f_post, double& g_post, double& sigma_f, double& sigma_g) {
    int N = X_buffer.size();
    if (N == 0) {
        f_post = f_prior; g_post = g_prior;
        sigma_f = var_f;  sigma_g = var_f;
        return;
    }

    Eigen::MatrixXd K_fg(N, N);
    Eigen::VectorXd Y_err(N);
    Eigen::VectorXd k_f(N), k_g(N);
    Eigen::MatrixXd U_mat = Eigen::MatrixXd::Zero(N, N);

    for (int i = 0; i < N; ++i) {
        U_mat(i, i) = U_buffer[i];
        k_f(i) = kernel_se(X_buffer[i], x_query);
        k_g(i) = kernel_se(X_buffer[i], x_query);
        
        double mu_i = F0_buffer[i] + U_buffer[i] * G0_buffer[i]; 
        Y_err(i) = Y_buffer[i] - mu_i;

        for (int j = 0; j <= i; ++j) {
            double kf_ij = kernel_se(X_buffer[i], X_buffer[j]);
            double kg_ij = kernel_se(X_buffer[i], X_buffer[j]);
            double val = kf_ij + U_buffer[i] * kg_ij * U_buffer[j];
            if (i == j) val += noise_var;
            K_fg(i, j) = val;
            K_fg(j, i) = val;
        }
    }

    Eigen::VectorXd K_inv_Y = K_fg.ldlt().solve(Y_err);
    Eigen::MatrixXd K_inv = K_fg.ldlt().solve(Eigen::MatrixXd::Identity(N, N));

    f_post = f_prior + k_f.transpose() * K_inv_Y;
    g_post = g_prior + (U_mat * k_g).transpose() * K_inv_Y;

    sigma_f = kernel_se(x_query, x_query) - k_f.transpose() * K_inv * k_f;
    sigma_g = kernel_se(x_query, x_query) - (U_mat * k_g).transpose() * K_inv * (U_mat * k_g);

    sigma_f = sqrt(max(0.0, sigma_f));
    sigma_g = sqrt(max(0.0, sigma_g));
}

// ==========================================
// UncertaintyAwarePID Implementation
// ==========================================
UncertaintyAwarePID::UncertaintyAwarePID(double kp, double kv, double alpha)
    : kp_(kp), kv_(kv), alpha_(alpha) {}

double UncertaintyAwarePID::compute(double err_p, double err_v, double ref_acc, 
                                    double sigma_f, double sigma_g, double u_prev) {
    double sigma_total = sigma_f + std::abs(u_prev) * sigma_g;
    double gain_scale = 1.0 + alpha_ * sigma_total;
    
    return ref_acc + gain_scale * (kp_ * err_p + kv_ * err_v);
}

// ==========================================
// RTMPC Implementation
// ==========================================
RTMPC::RTMPC(double dt, int H, const Eigen::Vector2d& Q_diag, double R, 
             const Eigen::Vector2d& Q_anc_diag, double R_anc,
             const Eigen::Vector2d& state_limit, const Eigen::Vector2d& input_limit)
    : dt_(dt), H_(H), Q_(Q_diag.asDiagonal()), R_(R), state_limit_(state_limit), input_limit_(input_limit),
      qp_solver_(H_, 2 * H_) 
{
    A_d << 1.0, dt_, 
           0.0, 1.0;
    B_d << 0.5 * dt_ * dt_, 
           dt_;

    compute_dlqr(Q_anc_diag.asDiagonal(), R_anc);
    
    qpOASES::Options options;
    options.printLevel = qpOASES::PL_NONE;
    qp_solver_.setOptions(options);
}

void RTMPC::compute_dlqr(const Eigen::Matrix2d& Q_anc, double R_anc) {
    Eigen::Matrix2d P = Q_anc;
    for (int i = 0; i < 100; ++i) {
        double R_eff = R_anc + (B_d.transpose() * P * B_d)(0, 0); 
        P = A_d.transpose() * P * A_d - 
            A_d.transpose() * P * B_d * (1.0 / R_eff) * B_d.transpose() * P * A_d + Q_anc;
    }
    double R_eff = R_anc + (B_d.transpose() * P * B_d)(0, 0);
    K_anc = (1.0 / R_eff) * B_d.transpose() * P * A_d;
}

double RTMPC::solve(const Eigen::Vector2d& current_state, const Eigen::Vector2d& ref_state, double w_kappa) {
    // 1. Compute Terminal Cost P using DARE (approximate with iterations)
    Eigen::Matrix2d P = Q_;
    for (int i = 0; i < 20; ++i) {
        double R_eff = R_ + (B_d.transpose() * P * B_d)(0, 0);
        P = A_d.transpose() * P * A_d - 
            A_d.transpose() * P * B_d * (1.0 / R_eff) * B_d.transpose() * P * A_d + Q_;
    }

    // 2. Setup QP Matrices
    int nV = H_;
    int nC = 2 * H_;
    
    Eigen::Vector2d x0 = current_state - ref_state;

    // Prediction matrices construction
    std::vector<Eigen::MatrixXd> A_pow(H_ + 1);
    A_pow[0] = Eigen::Matrix2d::Identity();
    for(int i=1; i<=H_; ++i) A_pow[i] = A_d * A_pow[i-1];

    Eigen::MatrixXd Mu = Eigen::MatrixXd::Zero(2 * (H_ + 1), H_);
    Eigen::MatrixXd Mx = Eigen::MatrixXd::Zero(2 * (H_ + 1), 2);

    for(int k=0; k<=H_; ++k) {
        Mx.block(2*k, 0, 2, 2) = A_pow[k];
    }
    for(int k=1; k<=H_; ++k) {
        for(int j=0; j<k; ++j) {
            Mu.block(2*k, j, 2, 1) = A_pow[k - 1 - j] * B_d;
        }
    }

    Eigen::MatrixXd Q_bar = Eigen::MatrixXd::Zero(2 * (H_ + 1), 2 * (H_ + 1));
    for(int k=0; k<H_; ++k) {
        Q_bar.block(2*k, 2*k, 2, 2) = Q_;
    }
    Q_bar.block(2*H_, 2*H_, 2, 2) = P;

    Eigen::MatrixXd R_bar = Eigen::MatrixXd::Identity(H_, H_) * R_;

    Eigen::MatrixXd H_qp = Mu.transpose() * Q_bar * Mu + R_bar;
    Eigen::VectorXd g_qp = Mu.transpose() * Q_bar * Mx * x0;

    // 3. Constraints
    Eigen::VectorXd lb = Eigen::VectorXd::Zero(nV);
    Eigen::VectorXd ub = Eigen::VectorXd::Zero(nV);
    Eigen::VectorXd lbA = Eigen::VectorXd::Zero(nC);
    Eigen::VectorXd ubA = Eigen::VectorXd::Zero(nC);

    // Input constraints (Tube tightened)
    double delta_u = (std::abs(K_anc(0)) + std::abs(K_anc(1))) * w_kappa;
    double u_lb = input_limit_(0) + delta_u;
    double u_ub = input_limit_(1) - delta_u;
    if (u_lb > u_ub) { // Safety clamp
        u_lb = input_limit_(0);
        u_ub = input_limit_(1);
    }
    for(int i=0; i<H_; ++i) {
        lb(i) = u_lb;
        ub(i) = u_ub;
    }

    // State constraints (Tube tightened)
    // A_qp * U <= ubA - Mx * x0
    Eigen::MatrixXd A_qp = Mu.block(2, 0, 2*H_, H_);
    Eigen::VectorXd Mx_x0 = Mx.block(2, 0, 2*H_, 2) * x0;

    for(int k=0; k<H_; ++k) {
        // Position
        lbA(2*k) = -state_limit_(0) + w_kappa - Mx_x0(2*k);
        ubA(2*k) = state_limit_(0) - w_kappa - Mx_x0(2*k);
        // Velocity
        lbA(2*k+1) = -state_limit_(1) + w_kappa - Mx_x0(2*k+1);
        ubA(2*k+1) = state_limit_(1) - w_kappa - Mx_x0(2*k+1);
    }

    // 4. Solve
    int nWSR = 100;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> H_qp_row = H_qp;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_qp_row = A_qp;

    qp_solver_.init(H_qp_row.data(), g_qp.data(), A_qp_row.data(), lb.data(), ub.data(), lbA.data(), ubA.data(), nWSR);

    Eigen::VectorXd u_opt(H_);
    qp_solver_.getPrimalSolution(u_opt.data());
    
    return u_opt(0);
}

// ==========================================
// Controller Implementation
// ==========================================
Controller::Controller(Parameter_t &param_) : param(param_) {
    Gravity = Eigen::Vector3d(0.0, 0.0, -param.gra);

    // Initialize GP
    gp_x = new OnlineGP(param.gp_l, param.gp_sigma_f, param.gp_beta, param.gp_N_max);
    gp_y = new OnlineGP(param.gp_l, param.gp_sigma_f, param.gp_beta, param.gp_N_max);
    gp_z = new OnlineGP(param.gp_l, param.gp_sigma_f, param.gp_beta, param.gp_N_max);

    // Initialize RTMPC
    mpc_x = new RTMPC(param.mpc_dt, param.mpc_H, 
                      Eigen::Vector2d(param.mpc_xy_Q_p, param.mpc_xy_Q_v), param.mpc_xy_R, 
                      Eigen::Vector2d(param.mpc_xy_Q_anc_p, param.mpc_xy_Q_anc_v), param.mpc_xy_R_anc, 
                      Eigen::Vector2d(param.mpc_xy_limit_p, param.mpc_xy_limit_v), 
                      Eigen::Vector2d(param.mpc_xy_limit_u_min, param.mpc_xy_limit_u_max));
                      
    mpc_y = new RTMPC(param.mpc_dt, param.mpc_H, 
                      Eigen::Vector2d(param.mpc_xy_Q_p, param.mpc_xy_Q_v), param.mpc_xy_R, 
                      Eigen::Vector2d(param.mpc_xy_Q_anc_p, param.mpc_xy_Q_anc_v), param.mpc_xy_R_anc, 
                      Eigen::Vector2d(param.mpc_xy_limit_p, param.mpc_xy_limit_v), 
                      Eigen::Vector2d(param.mpc_xy_limit_u_min, param.mpc_xy_limit_u_max));

    mpc_z = new RTMPC(param.mpc_dt, param.mpc_H, 
                      Eigen::Vector2d(param.mpc_z_Q_p, param.mpc_z_Q_v), param.mpc_z_R, 
                      Eigen::Vector2d(param.mpc_z_Q_anc_p, param.mpc_z_Q_anc_v), param.mpc_z_R_anc, 
                      Eigen::Vector2d(param.mpc_z_limit_p, param.mpc_z_limit_v), 
                      Eigen::Vector2d(param.mpc_z_limit_u_min, param.mpc_z_limit_u_max));

    // Initialize Uncertainty-Aware PID
    pid_x = new UncertaintyAwarePID(param.gain.Kp0, param.gain.Kv0, param.ua_pid_alpha);
    pid_y = new UncertaintyAwarePID(param.gain.Kp1, param.gain.Kv1, param.ua_pid_alpha);
    pid_z = new UncertaintyAwarePID(param.gain.Kp2, param.gain.Kv2, param.ua_pid_alpha);

    last_u_x = 0; last_u_y = 0; last_u_z = 0;

    // Load LibTorch Models
    try {
        // NOTE: Ensure these .pt files are in the directory where the ROS node is executed
        // or provide absolute paths.
        prior_model_x = torch::jit::load("generator_prior_X.pt");
        prior_model_y = torch::jit::load("generator_prior_Y.pt");
        prior_model_z = torch::jit::load("generator_prior_Z.pt");
        models_loaded = true;
        std::cout << "[Controller] LibTorch prior models loaded successfully.\n";
    } catch (const c10::Error& e) {
        std::cerr << "[Controller] Warning: Could not load LibTorch models. Using 0/1 prior.\n";
        models_loaded = false;
    }
}

void Controller::get_prior(double input_val, char axis, double& f0, double& g0) {
    if (!models_loaded) {
        f0 = 0.0;
        g0 = 1.0;
        return;
    }

    // Pass the input value directly to the 1D network
    torch::Tensor input_tensor = torch::tensor({{static_cast<float>(input_val)}}); 
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    torch::jit::script::Module* target_module;
    if (axis == 'X') target_module = &prior_model_x;
    else if (axis == 'Y') target_module = &prior_model_y;
    else target_module = &prior_model_z;

    // Forward inference
    auto output = target_module->forward(inputs).toTuple();
    f0 = output->elements()[0].toTensor().item<float>();
    g0 = output->elements()[1].toTensor().item<float>();
}

quadrotor_msgs::Px4ctrlDebug Controller::update(
    const Desired_State_t &des,
    const Odom_Data_t &odom,
    const Imu_Data_t &imu,
    Controller_Output_t &u,
    double voltage) 
{
    Eigen::Vector2d state_x(odom.p(0), odom.v(0));
    Eigen::Vector2d state_y(odom.p(1), odom.v(1));
    Eigen::Vector2d state_z(odom.p(2), odom.v(2));

    Eigen::Vector2d ref_x(des.p(0), des.v(0));
    Eigen::Vector2d ref_y(des.p(1), des.v(1));
    Eigen::Vector2d ref_z(des.p(2), des.v(2));

    Eigen::Vector3d obs_acc = imu.a - Gravity;

    // Get Neural Network Priors
    double f0_x, g0_x; get_prior(obs_acc(0), 'X', f0_x, g0_x);
    double f0_y, g0_y; get_prior(obs_acc(1), 'Y', f0_y, g0_y);
    double f0_z, g0_z; get_prior(obs_acc(2), 'Z', f0_z, g0_z);

    gp_x->add_data(state_x, last_u_x, obs_acc(0), f0_x, g0_x);
    gp_y->add_data(state_y, last_u_y, obs_acc(1), f0_y, g0_y);
    gp_z->add_data(state_z, last_u_z, obs_acc(2), f0_z, g0_z);

    // Get GP Posteriors
    double fx_post, gx_post, sig_fx, sig_gx;
    double fy_post, gy_post, sig_fy, sig_gy;
    double fz_post, gz_post, sig_fz, sig_gz;

    gp_x->predict(state_x, last_u_x, f0_x, g0_x, fx_post, gx_post, sig_fx, sig_gx);
    gp_y->predict(state_y, last_u_y, f0_y, g0_y, fy_post, gy_post, sig_fy, sig_gy);
    gp_z->predict(state_z, last_u_z, f0_z, g0_z, fz_post, gz_post, sig_fz, sig_gz);

    double beta = param.gp_beta;
    double w_kappa_x = beta * sig_fx + abs(last_u_x) * beta * sig_gx;
    double w_kappa_y = beta * sig_fy + abs(last_u_y) * beta * sig_gy;
    double w_kappa_z = beta * sig_fz + abs(last_u_z) * beta * sig_gz;

    double eta_x, eta_y, eta_z;

    if (param.use_mpc) {
        double bar_eta_x = mpc_x->solve(state_x, ref_x, w_kappa_x);
        double bar_eta_y = mpc_y->solve(state_y, ref_y, w_kappa_y);
        double bar_eta_z = mpc_z->solve(state_z, ref_z, w_kappa_z);

        eta_x = bar_eta_x + des.a(0);
        eta_y = bar_eta_y + des.a(1);
        eta_z = bar_eta_z + des.a(2);
    } else {
        // Uncertainty-Aware PID
        // Error definition: e = x_des - x (Standard control convention often uses e = x_des - x for P term)
        // But RTMPC code uses state - ref. Let's stick to standard P control: Kp * (ref - state)
        // The paper says eta = -K * e + x_nd_dot. If e = x - x_ref, then -K(x-x_ref) is correct.
        eta_x = pid_x->compute(ref_x(0) - state_x(0), ref_x(1) - state_x(1), des.a(0), sig_fx, sig_gx, last_u_x);
        eta_y = pid_y->compute(ref_y(0) - state_y(0), ref_y(1) - state_y(1), des.a(1), sig_fy, sig_gy, last_u_y);
        eta_z = pid_z->compute(ref_z(0) - state_z(0), ref_z(1) - state_z(1), des.a(2), sig_fz, sig_gz, last_u_z);
    }

    double u_cmd_x = (eta_x - fx_post) / max(0.1, gx_post);
    double u_cmd_y = (eta_y - fy_post) / max(0.1, gy_post);
    double u_cmd_z = (eta_z - fz_post) / max(0.1, gz_post);

    last_u_x = u_cmd_x; last_u_y = u_cmd_y; last_u_z = u_cmd_z;

    Eigen::Vector3d thr_acc(u_cmd_x, u_cmd_y, u_cmd_z + param.gra); 
    
    Eigen::Quaterniond desired_attitude;
    computeFlatInput(thr_acc, des.yaw, odom.q, desired_attitude, u.thrust);

    u.q = imu.q * odom.q.inverse() * desired_attitude; 
    u.bodyrates = Eigen::Vector3d::Zero();

    debug.des_a_x = thr_acc(0);
    debug.des_a_y = thr_acc(1);
    debug.des_a_z = thr_acc(2);
    return debug;
}

void Controller::computeFlatInput(const Eigen::Vector3d &thr_acc,
                                  const double &yaw,
                                  const Eigen::Quaterniond &att_est,
                                  Eigen::Quaterniond &att,
                                  double &thrust) const 
{
    if (thr_acc.norm() < 1e-4) {
        att = att_est;
        thrust = param.mass * param.gra; 
        return;
    }
    Eigen::Vector3d zb = thr_acc.normalized();
    Eigen::Vector3d xc(cos(yaw), sin(yaw), 0.0);
    Eigen::Vector3d yc = zb.cross(xc).normalized();
    Eigen::Vector3d xb = yc.cross(zb).normalized();
    Eigen::Matrix3d R;
    R << xb, yc, zb;
    att = Eigen::Quaterniond(R);
    
    thrust = thr_acc.dot(zb) / (param.gra / 0.04); 
}

void Controller::resetThrustMapping() {}

bool Controller::estimateThrustModel(const Eigen::Vector3d &est_a, double voltage, const Eigen::Vector3d &est_v, const Parameter_t &param) {
    return true;
}