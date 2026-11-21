#include <ament_index_cpp/get_package_share_directory.hpp>
#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <iostream>
#include <stdexcept>

using namespace ocs2;
using namespace ocs2::humanoid;

int main() {
  const auto config = ament_index_cpp::get_package_share_directory("g1_centroidal_mpc");
  const auto urdf = ament_index_cpp::get_package_share_directory("g1_description") + "/urdf/g1_29dof.urdf";
  const auto reference = config + "/config/command/reference_random_reference.info";
  CentroidalMpcInterface interface(config + "/config/mpc/task_random_reference.info", urdf, reference, false);
  auto pin = interface.getPinocchioInterface();
  const auto& info = interface.getCentroidalModelInfo();
  const auto& robot = interface.getMpcRobotModel();
  auto state = interface.getInitialState();
  updateCentroidalDynamics(pin, info, robot.getGeneralizedCoordinates(state));
  const matrix_t Ab = getCentroidalMomentumMatrix(pin).leftCols<6>();
  vector6_t yawVelocity = vector6_t::Zero();
  yawVelocity[3] = 0.4;
  const vector6_t yawMomentum = Ab * yawVelocity / info.robotMass;
  state.head<6>() = yawMomentum;
  state[0] = 0.35;
  state[1] = -0.1;
  WalkingVelocityCommand command;
  command.linear_velocity_x = 0.7;
  command.linear_velocity_y = 0.1;
  command.angular_velocity_z = 0.4;
  command.desired_pelvis_height = state[8];
  CentroidalMpcTargetTrajectoriesCalculator calculator(reference, robot, pin, info, 1.2);
  const auto smooth = calculator.commandedVelocityToTargetTrajectories(command, 0.0, state);
  if (std::abs(smooth.stateTrajectory.front()[0] - 0.35) > 1e-10 ||
      std::abs(smooth.stateTrajectory.front()[1] + 0.1) > 1e-10 ||
      (smooth.stateTrajectory.front().segment<3>(3) - yawMomentum.tail<3>()).norm() > 1e-8) {
    throw std::runtime_error("Normalized momentum or yaw-index regression");
  }
  calculator.setVelocityReferenceSmoothingEnabled(false);
  const auto direct = calculator.commandedVelocityToTargetTrajectories(command, 0.0, state);
  for (const auto& target : direct.stateTrajectory) {
    if (std::abs(target[0] - 0.7) > 1e-10 || std::abs(target[1] - 0.1) > 1e-10 ||
        (target.segment<3>(3) - yawMomentum.tail<3>()).norm() > 1e-8) {
      throw std::runtime_error("Conditioned command was re-filtered or re-ramped");
    }
  }
  std::cout << "Velocity reference regression checks passed\n";
}
