#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include "Eigen/src/Core/util/ForwardDeclarations.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

MatrixXd CreateSigmaPoints(const VectorXd& x, const MatrixXd& P, const MatrixXd& Q, double lambda) {
  int n_x = x.rows();
  int n_aug_x = Q.rows() + n_x;

  VectorXd x_aug = VectorXd::Zero(n_aug_x);
  x_aug.head(n_x) = x;

  MatrixXd P_aug = MatrixXd::Zero(n_aug_x, n_aug_x);
  P_aug.topLeftCorner(n_x, n_x) = P;
  P_aug.bottomRightCorner(n_aug_x - n_x, n_aug_x - n_x) = Q;

  MatrixXd result(n_aug_x, 2 * n_aug_x + 1);
  MatrixXd A = P_aug.llt().matrixL();

  result.col(0) = x_aug;

  for (int i = 0; i < n_aug_x; i++) {
    result.col(i + 1) = x_aug + sqrt(lambda + n_aug_x) * A.col(i);
    result.col(i + 1 + n_aug_x) = x_aug - sqrt(lambda + n_aug_x) * A.col(i);
  }

  return result;
}

MatrixXd PredictSigmaPoints(const MatrixXd& augmented_sigma_points, double delta_t) {
  MatrixXd result = MatrixXd::Zero(5, augmented_sigma_points.cols());

  for (int i = 0; i < augmented_sigma_points.cols(); ++i) {
    //extract values for better readability
    double p_x = augmented_sigma_points(0, i);
    double p_y = augmented_sigma_points(1, i);
    double v = augmented_sigma_points(2, i);
    double yaw = augmented_sigma_points(3, i);
    double yawd = augmented_sigma_points(4, i);
    double nu_a = augmented_sigma_points(5, i);
    double nu_yawdd = augmented_sigma_points(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    result(0, i) = px_p;
    result(1, i) = py_p;
    result(2, i) = v_p;
    result(3, i) = yaw_p;
    result(4, i) = yawd_p;
  }

  return result;
}

pair<VectorXd, MatrixXd> PredictMeanAndCovariance(const MatrixXd& predicted_sigma_points, double lambda, int n_aug) {
  int n_x = predicted_sigma_points.rows();
  int cols = predicted_sigma_points.cols();
  VectorXd x = VectorXd::Zero(n_x);
  MatrixXd P = MatrixXd::Zero(n_x, n_x);
  VectorXd w = VectorXd::Zero(cols);

  w(0) = lambda / (lambda + n_aug);
  for (int i = 1; i < cols; ++i) {
    w(i) = 0.5 / (lambda + n_aug);
  }

  for (int i = 0; i < cols; ++i) {
    x += w(i) * predicted_sigma_points.col(i);
  }

  for (int i = 0; i < cols; ++i) {
    VectorXd x_diff = predicted_sigma_points.col(i) - x;

    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P = P + w(i) * x_diff * x_diff.transpose();
  }

  return make_pair(x, P);
}


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() : is_initialized_(false),
             x_(VectorXd::Zero(n_x_)),
             P_(1000 * MatrixXd::Identity(n_x_, n_x_)),
             time_us_(0),
             lambda_(3 - n_x_),
             NIS_radar_(0),
             NIS_laser_(0),
             Q_(MatrixXd::Zero(n_aug_ - n_x_, n_aug_ - n_x_)),
             previous_timestamp_(0) {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  Q_(0, 0) = std_a_ * std_a_;
  Q_(1, 1) = std_yawdd_ * std_yawdd_;

  /**
TODO:

Complete the initialization. See ukf.h for other member properties.

Hint: one or more values initialized above might be wildly off...
*/
}

void UKF::Initialize(const MeasurementPackage& measurement_pack) {
  // first measurement
  previous_timestamp_ = measurement_pack.timestamp_;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    double rho = measurement_pack.raw_measurements_(0);
    double phi = measurement_pack.raw_measurements_(1);
    x_(0) = rho * cos(phi);
    x_(1) = rho * sin(phi);
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    x_(0) = measurement_pack.raw_measurements_(0);
    x_(1) = measurement_pack.raw_measurements_(1);
  }

  is_initialized_ = true;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {
  if (!is_initialized_) {
    Initialize(meas_package);
    return;
  }

  double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = meas_package.timestamp_;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  else {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd augmented_sigma_points = CreateSigmaPoints(x_, P_, Q_, lambda_);
  MatrixXd predicted_sigma_points = PredictSigmaPoints(augmented_sigma_points, delta_t);

  auto x_and_P = PredictMeanAndCovariance(predicted_sigma_points, lambda_, n_aug_);
  x_ = x_and_P.first;
  P_ = x_and_P.second;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
