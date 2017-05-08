#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

void NormalizeAngles(VectorXd & v, int angleNormalizationIndex) {
  if (angleNormalizationIndex < 0) return;

  while (v(angleNormalizationIndex) > M_PI) v(angleNormalizationIndex) -= 2. * M_PI;
  while (v(angleNormalizationIndex) < -M_PI) v(angleNormalizationIndex) += 2. * M_PI;
}

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

pair<VectorXd, MatrixXd> ComputeMeanAndCovariance(const MatrixXd& predicted_sigma_points, const VectorXd& weights, int angleNormalizationIndex) {
  int n_x = predicted_sigma_points.rows();
  int cols = predicted_sigma_points.cols();
  VectorXd x = VectorXd::Zero(n_x);
  MatrixXd P = MatrixXd::Zero(n_x, n_x);

  for (int i = 0; i < cols; ++i) {
    x += weights(i) * predicted_sigma_points.col(i);
  }

  for (int i = 0; i < cols; ++i) {
    VectorXd x_diff = predicted_sigma_points.col(i) - x;

    NormalizeAngles(x_diff, angleNormalizationIndex);

    P = P + weights(i) * x_diff * x_diff.transpose();
  }

  return make_pair(x, P);
}

MatrixXd SigmaPointsToRadarSpace(const MatrixXd& predicted_sigma_points) {
  MatrixXd result = MatrixXd::Zero(3, predicted_sigma_points.cols());
  for (int i = 0; i < predicted_sigma_points.cols(); i++) {

    // extract values for better readibility
    double p_x = predicted_sigma_points(0, i);
    double p_y = predicted_sigma_points(1, i);
    double v = predicted_sigma_points(2, i);
    double yaw = predicted_sigma_points(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    result(0, i) = sqrt(p_x * p_x + p_y * p_y); //r
    result(1, i) = atan2(p_y, p_x); //phi
    result(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
  }

  return result;
}

MatrixXd ComputeCrossCorrelation(const VectorXd& x, const MatrixXd& Xsig_pred, const VectorXd& z, const MatrixXd& Zsig, const VectorXd& weights, int angleNormalizationIndex) {
  MatrixXd result = MatrixXd::Zero(x.rows(), z.rows());

  for (int i = 0; i < Xsig_pred.cols(); i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z;

    NormalizeAngles(z_diff, angleNormalizationIndex);

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;

    NormalizeAngles(x_diff, 3);

    result += weights(i) * x_diff * z_diff.transpose();
  }

  return result;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() : x_(VectorXd::Zero(n_x_)),
             P_(MatrixXd::Identity(n_x_, n_x_)),
             NIS_radar_(0),
             NIS_laser_(0),
             is_initialized_(false),
             weights_(VectorXd::Zero(n_aug_ * 2 + 1)),
             lambda_(3 - n_x_),
             Q_(MatrixXd::Zero(n_aug_ - n_x_, n_aug_ - n_x_)),
             R_radar_(MatrixXd::Zero(3, 3)),
             R_laser_(MatrixXd::Zero(2, 2)),
             previous_timestamp_(0) {
  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_ = 1;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  double std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ = 0.3;

  Q_(0, 0) = std_a_ * std_a_;
  Q_(1, 1) = std_yawdd_ * std_yawdd_;

  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < weights_.size(); ++i) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  R_radar_(0, 0) = std_radr_ * std_radr_;
  R_radar_(1, 1) = std_radphi_ * std_radphi_;
  R_radar_(2, 2) = std_radrd_ * std_radrd_;

  R_laser_(0, 0) = std_laspx_ * std_laspx_;
  R_laser_(1, 1) = std_laspy_ * std_laspy_;
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
  Xsig_pred_ = PredictSigmaPoints(augmented_sigma_points, delta_t);

  auto x_and_P = ComputeMeanAndCovariance(Xsig_pred_, weights_, 3);
  x_ = x_and_P.first;
  P_ = x_and_P.second;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  MatrixXd Zsig = Xsig_pred_.topRows(2);
  VectorXd z_pred = x_.head(2);
  MatrixXd S = P_.topLeftCorner(2, 2) + R_laser_;
  MatrixXd T = ComputeCrossCorrelation(x_, Xsig_pred_, z_pred, Zsig, weights_, -1);

  MatrixXd K = T * S.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  MatrixXd Zsig = SigmaPointsToRadarSpace(Xsig_pred_);
  auto z_and_S = ComputeMeanAndCovariance(Zsig, weights_, 1);
  VectorXd z_pred = z_and_S.first;
  MatrixXd S = z_and_S.second + R_radar_;
  MatrixXd T = ComputeCrossCorrelation(x_, Xsig_pred_, z_pred, Zsig, weights_, 1);

  MatrixXd K = T * S.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  NormalizeAngles(z_diff, 1);

  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
