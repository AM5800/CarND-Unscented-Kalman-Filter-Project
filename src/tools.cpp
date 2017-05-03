#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  if (estimations.size() != ground_truth.size() || estimations.size() == 0)
    throw std::exception("Estimation and Ground Truth size mismatch");

  VectorXd sum = VectorXd::Zero(estimations.front().rows());

  for (size_t i = 0; i < estimations.size(); ++i) {
    VectorXd residual = ground_truth[i] - estimations[i];
    VectorXd residual_squared = residual.array() * residual.array();
    sum += residual_squared;
  }

  return (sum / estimations.size()).array().sqrt();
}
