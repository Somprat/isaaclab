/******************************************************************************
Torso-origin height calibration for the G1 centroidal MPC.

Given a requested torso_link frame-origin world height, repeatedly runs a
standing (double-support, zero-velocity) MPC rollout with a pelvis height
command, measures the settled torso_link origin height on the full robot model and
corrects the pelvis command until the torso height matches. The settled
whole-robot CoM is then saved for the LIPM reference generator.

The rollout is the MPC policy rollout used by the random reference generator
(CentroidalMpcRandomReferenceGenerator.cpp); it is not a physics simulation.
******************************************************************************/

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>
#include <ocs2_sqp/SqpMpc.h>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include <humanoid_centroidal_mpc_ros2/FrameHeightCost.h>
#include <humanoid_common_mpc/gait/GaitScheduleUpdater.h>
#include <humanoid_common_mpc/gait/ModeSequenceTemplate.h>
#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/pinocchio_model/createPinocchioModel.h>

#include <urdf/model.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ocs2;
using namespace ocs2::humanoid;

namespace {

constexpr const char* kTorsoLink = "torso_link";

struct Options {
  std::string taskFile;
  std::string referenceFile;
  std::string urdfFile;
  std::string gaitFile;
  std::string output = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/generated_motions/g1_height_calibration.json";
  scalar_t torsoOriginHeight = std::numeric_limits<scalar_t>::quiet_NaN();
  scalar_t tolerance = 1e-3;          // [m] torso origin height match tolerance
  size_t maxIterations = 25;
  scalar_t gain = 0.8;                // damped secant (Newton) step factor
  scalar_t maxCorrection = 0.05;      // [m] per-iteration pelvis command step
  scalar_t minSlope = 0.02;           // bounds on the estimated d(torso z)/d(pelvis command)
  scalar_t maxSlope = 2.0;
  scalar_t minPelvisHeight = 0.30;    // [m] command bounds
  scalar_t maxPelvisHeight = 0.85;
  scalar_t dt = 0.02;                 // [s] MPC rollout step
  scalar_t minSettleTime = 1.0;       // [s] per iteration before settling is checked
  scalar_t maxSettleTime = 10.0;      // [s] per iteration
  scalar_t settleWindow = 0.5;        // [s] window over which motion must be low
  scalar_t settleHeightSpan = 2e-4;   // [m] max torso origin z span over the window
  scalar_t settleComSpeed = 2e-3;     // [m/s] max whole-body CoM speed over the window
  scalar_t settleJointSpeed = 1e-2;   // [rad/s] max joint speed over the window
  scalar_t torsoHeightWeight = 0.0;   // > 0 adds a torso_link origin height cost at the requested height
};

std::string requireValue(int& i, int argc, char** argv) {
  if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + argv[i]);
  return argv[++i];
}

scalar_t parseScalar(const std::string& text, const std::string& option) {
  size_t used = 0;
  const scalar_t value = std::stod(text, &used);
  if (used != text.size() || !std::isfinite(value)) throw std::runtime_error("Invalid value for " + option + ": " + text);
  return value;
}

void printUsage() {
  std::cout << "Usage: humanoid_centroidal_mpc_height_calibration --torso-origin-height Z [options]\n"
               "  --output PATH             JSON result (written on success and failure)\n"
               "  --task-file/--reference-file/--urdf-file/--gait-file PATH\n"
               "  --tolerance M (1e-3)  --max-iterations N (25)  --gain G (0.8)  --max-correction M (0.05)\n"
               "  --min-slope S (0.02)  --max-slope S (2.0)\n"
               "  --min-pelvis-height M (0.30)  --max-pelvis-height M (0.85)  --dt S (0.02)\n"
               "  --min-settle-time S (1.0)  --max-settle-time S (10.0)  --settle-window S (0.5)\n"
               "  --settle-height-span M (2e-4)  --settle-com-speed M/S (2e-3)  --settle-joint-speed RAD/S (1e-2)\n";
}

Options parseOptions(int argc, char** argv) {
  Options o;
  const auto g1MpcShare = ament_index_cpp::get_package_share_directory("g1_centroidal_mpc");
  const auto g1DescriptionShare = ament_index_cpp::get_package_share_directory("g1_description");
  const auto commonMpcShare = ament_index_cpp::get_package_share_directory("humanoid_common_mpc");
  o.taskFile = g1MpcShare + "/config/mpc/task_random_reference.info";
  o.referenceFile = g1MpcShare + "/config/command/reference_random_reference.info";
  o.urdfFile = g1DescriptionShare + "/urdf/g1_29dof.urdf";
  o.gaitFile = commonMpcShare + "/config/command/gait.info";

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      printUsage();
      std::exit(0);
    } else if (arg == "--torso-origin-height") o.torsoOriginHeight = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--output") o.output = requireValue(i, argc, argv);
    else if (arg == "--torso-height-weight") o.torsoHeightWeight = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--task-file") o.taskFile = requireValue(i, argc, argv);
    else if (arg == "--reference-file") o.referenceFile = requireValue(i, argc, argv);
    else if (arg == "--urdf-file") o.urdfFile = requireValue(i, argc, argv);
    else if (arg == "--gait-file") o.gaitFile = requireValue(i, argc, argv);
    else if (arg == "--tolerance") o.tolerance = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--max-iterations") o.maxIterations = static_cast<size_t>(parseScalar(requireValue(i, argc, argv), arg));
    else if (arg == "--gain") o.gain = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--max-correction") o.maxCorrection = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--min-slope") o.minSlope = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--max-slope") o.maxSlope = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--min-pelvis-height") o.minPelvisHeight = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--max-pelvis-height") o.maxPelvisHeight = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--dt") o.dt = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--min-settle-time") o.minSettleTime = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--max-settle-time") o.maxSettleTime = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--settle-window") o.settleWindow = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--settle-height-span") o.settleHeightSpan = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--settle-com-speed") o.settleComSpeed = parseScalar(requireValue(i, argc, argv), arg);
    else if (arg == "--settle-joint-speed") o.settleJointSpeed = parseScalar(requireValue(i, argc, argv), arg);
    else throw std::runtime_error("Unknown option: " + arg);
  }
  if (!std::isfinite(o.torsoOriginHeight) || o.torsoOriginHeight <= 0) throw std::runtime_error("--torso-origin-height Z (> 0) is required.");
  if (o.tolerance <= 0 || o.maxIterations < 1 || o.gain <= 0 || o.gain > 1 || o.maxCorrection <= 0)
    throw std::runtime_error("Require tolerance > 0, max-iterations >= 1, gain in (0, 1], max-correction > 0.");
  if (!(o.minSlope > 0 && o.minSlope < o.maxSlope)) throw std::runtime_error("Require 0 < min-slope < max-slope.");
  if (!(o.minPelvisHeight > 0 && o.minPelvisHeight < o.maxPelvisHeight)) throw std::runtime_error("Require 0 < min-pelvis-height < max.");
  if (o.dt <= 0 || o.settleWindow < o.dt || o.minSettleTime < o.settleWindow || o.maxSettleTime < o.minSettleTime)
    throw std::runtime_error("Require dt > 0 and dt <= settle-window <= min-settle-time <= max-settle-time.");
  if (!std::isfinite(o.torsoHeightWeight) || o.torsoHeightWeight < 0) throw std::runtime_error("--torso-height-weight must be >= 0.");
  if (o.settleHeightSpan <= 0 || o.settleComSpeed <= 0 || o.settleJointSpeed <= 0) throw std::runtime_error("Settle thresholds must be positive.");
  return o;
}

/// Full-model kinematic measurements for one MPC state.
struct Posture {
  vector6_t basePose;      // [x, y, z, yaw, pitch, roll]
  vector_t fullJointPos;
  vector3_t torsoLinkPos;  // torso_link frame origin, world
  vector3_t torsoComPos;   // torso_link inertial CoM, world
  vector3_t wholeBodyCom;  // whole-robot CoM, world
  std::vector<vector3_t> contactPos;
};

class FullModelProbe {
 public:
  explicit FullModelProbe(const std::string& urdfFile)
      : interface_(createDefaultPinocchioInterface(urdfFile)), model_(interface_.getModel()), data_(model_) {
    torsoFrame_ = frameId(kTorsoLink);
    // Only torso_link's own inertial origin: Pinocchio merges fixed children (head, IMU, ...) into the joint body.
    urdf::Model urdfModel;
    if (!urdfModel.initFile(urdfFile)) throw std::runtime_error("Failed to parse URDF: " + urdfFile);
    const auto link = urdfModel.getLink(kTorsoLink);
    if (!link || !link->inertial) throw std::runtime_error("URDF torso_link has no inertial element.");
    const auto& p = link->inertial->origin.position;
    torsoInertialOffset_ << p.x, p.y, p.z;
    torsoMass_ = link->inertial->mass;
    totalMass_ = pinocchio::computeTotalMass(model_);
  }

  Posture measure(const vector6_t& basePose, const vector_t& fullJointPos) {
    vector_t q = vector_t::Zero(model_.nq);
    q.head<6>() = basePose;  // Same root layout as the reference generator's appendBodyPoses.
    q.tail(fullJointPos.size()) = fullJointPos;
    pinocchio::forwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);
    Posture posture;
    posture.basePose = basePose;
    posture.fullJointPos = fullJointPos;
    const auto& torso = data_.oMf[torsoFrame_];
    posture.torsoLinkPos = torso.translation();
    posture.torsoComPos = torso.act(torsoInertialOffset_);  // R * offset + p
    posture.wholeBodyCom = pinocchio::centerOfMass(model_, data_, q, false);
    return posture;
  }

  const vector3_t& torsoInertialOffset() const { return torsoInertialOffset_; }
  scalar_t torsoMass() const { return torsoMass_; }
  scalar_t totalMass() const { return totalMass_; }

 private:
  pinocchio::FrameIndex frameId(const std::string& name) const {
    const auto id = model_.getFrameId(name);
    if (id == static_cast<pinocchio::FrameIndex>(model_.nframes)) throw std::runtime_error("Frame not found in full model: " + name);
    return id;
  }

  PinocchioInterface interface_;
  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::FrameIndex torsoFrame_;
  vector3_t torsoInertialOffset_;
  scalar_t torsoMass_ = 0.0;
  scalar_t totalMass_ = 0.0;
};

struct SettleResult {
  bool settled = false;
  scalar_t duration = 0.0;
  scalar_t heightSpan = 0.0;
  scalar_t maxComSpeed = 0.0;
  scalar_t maxJointSpeed = 0.0;
  bool doubleSupport = true;
};

struct IterationRecord {
  size_t iteration;
  scalar_t pelvisCommand;
  scalar_t measuredPelvis;
  scalar_t torsoOriginZ;
  scalar_t error;
  scalar_t slope;
  SettleResult settle;
};

std::string jsonVector(const vector_t& v) {
  std::ostringstream ss;
  ss << std::setprecision(10) << "[";
  for (Eigen::Index i = 0; i < v.size(); ++i) ss << (i ? ", " : "") << v[i];
  ss << "]";
  return ss.str();
}

std::string jsonString(const std::string& s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  return out + "\"";
}

void writeResult(const Options& o, const FullModelProbe& probe, const std::vector<std::string>& jointNames,
                 const std::vector<std::string>& contactNames, bool success, const std::string& status,
                 const std::vector<IterationRecord>& history, const Posture* posture, scalar_t pelvisCommand, scalar_t initialPelvis,
                 scalar_t initialTorso) {
  std::filesystem::path path(o.output);
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::ofstream f(path);
  if (!f) throw std::runtime_error("Cannot write " + o.output);
  f << std::setprecision(10);
  f << "{\n  \"schema_version\": 2,\n  \"kind\": \"g1_torso_origin_height_calibration\",\n";
  f << "  \"success\": " << (success ? "true" : "false") << ",\n  \"status\": " << jsonString(status) << ",\n";
  f << "  \"frame\": \"world_z_up\",\n  \"source\": \"wb_humanoid_mpc standing MPC policy rollout (not physics)\",\n";
  f << "  \"urdf_file\": " << jsonString(o.urdfFile) << ",\n  \"task_file\": " << jsonString(o.taskFile) << ",\n";
  f << "  \"reference_file\": " << jsonString(o.referenceFile) << ",\n";
  f << "  \"requested_torso_origin_height\": " << o.torsoOriginHeight << ",\n  \"tolerance\": " << o.tolerance << ",\n";
  f << "  \"torso_height_cost_weight\": " << o.torsoHeightWeight << ",\n";
  f << "  \"torso_body_name\": \"torso_link\",\n  \"torso_inertial_offset\": " << jsonVector(probe.torsoInertialOffset()) << ",\n";
  f << "  \"torso_link_mass\": " << probe.torsoMass() << ",\n  \"total_mass_kg\": " << probe.totalMass() << ",\n";
  f << "  \"initial_pelvis_height\": " << initialPelvis << ",\n  \"initial_torso_origin_height\": " << initialTorso << ",\n";
  f << "  \"pelvis_height_command\": " << pelvisCommand << ",\n";
  if (posture != nullptr) {
    scalar_t ground = 0.0;
    for (const auto& p : posture->contactPos) ground += p.z();
    ground /= static_cast<scalar_t>(posture->contactPos.size());
    f << "  \"measured_pelvis_height\": " << posture->basePose[2] << ",\n";
    f << "  \"base_pose_xyz_yaw_pitch_roll\": " << jsonVector(posture->basePose) << ",\n";
    f << "  \"torso_link_pos\": " << jsonVector(posture->torsoLinkPos) << ",\n";
    f << "  \"torso_com_pos\": " << jsonVector(posture->torsoComPos) << ",\n";
    f << "  \"measured_torso_origin_height\": " << posture->torsoLinkPos.z() << ",\n";
    f << "  \"torso_com_height\": " << posture->torsoComPos.z() << ",\n";
    f << "  \"whole_body_com_pos\": " << jsonVector(posture->wholeBodyCom) << ",\n";
    f << "  \"whole_body_com_height\": " << posture->wholeBodyCom.z() << ",\n";
    f << "  \"mean_contact_frame_height\": " << ground << ",\n";
    f << "  \"contact_names\": [";
    for (size_t i = 0; i < contactNames.size(); ++i) f << (i ? ", " : "") << jsonString(contactNames[i]);
    f << "],\n  \"contact_frame_pos\": [";
    for (size_t i = 0; i < posture->contactPos.size(); ++i) f << (i ? ", " : "") << jsonVector(posture->contactPos[i]);
    f << "],\n  \"joint_names\": [";
    for (size_t i = 0; i < jointNames.size(); ++i) f << (i ? ", " : "") << jsonString(jointNames[i]);
    f << "],\n  \"joint_pos\": " << jsonVector(posture->fullJointPos) << ",\n";
  }
  f << "  \"iterations\": [\n";
  for (size_t i = 0; i < history.size(); ++i) {
    const auto& h = history[i];
    f << "    {\"iteration\": " << h.iteration << ", \"pelvis_command\": " << h.pelvisCommand << ", \"measured_pelvis_height\": "
      << h.measuredPelvis << ", \"torso_origin_height\": " << h.torsoOriginZ << ", \"error\": " << h.error << ", \"slope_used\": " << h.slope
      << ", \"settled\": " << (h.settle.settled ? "true" : "false") << ", \"settle_time\": " << h.settle.duration
      << ", \"torso_height_span\": " << h.settle.heightSpan << ", \"max_com_speed\": " << h.settle.maxComSpeed
      << ", \"max_joint_speed\": " << h.settle.maxJointSpeed << ", \"double_support\": " << (h.settle.doubleSupport ? "true" : "false")
      << "}" << (i + 1 < history.size() ? "," : "") << "\n";
  }
  f << "  ]\n}\n";
}

int run(const Options& o) {
  CentroidalMpcInterface interface(o.taskFile, o.urdfFile, o.referenceFile, true);
  interface.getSwitchedModelReferenceManagerPtr()->setArmSwingReferenceActive(false);
  if (o.torsoHeightWeight > 0) {
    // Direct torso height objective; the pelvis command alone cannot overcome the MPC's upward bias below ~0.77 m.
    interface.getOptimalControlProblemRef().stateCostPtr->add(
        "torsoOriginHeightCost", std::make_unique<FrameHeightCost>(interface.getPinocchioInterface(), interface.getMpcRobotModel(),
                                                                   "torso_link", o.torsoOriginHeight, o.torsoHeightWeight));
    std::cout << "[height_calibration] torso_link origin height cost: target " << o.torsoOriginHeight << " m, weight "
              << o.torsoHeightWeight << "\n";
  }

  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(), interface.getInitializer());
  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  MPC_MRT_Interface mrt(mpc);
  mrt.initRollout(&interface.getRollout());

  const auto& mpcRobotModel = interface.getMpcRobotModel();
  const auto& modelSettings = interface.modelSettings();

  scalar_t defaultBaseHeight = 0.0;
  vector_t defaultMpcJointState(mpcRobotModel.getJointDim());
  loadData::loadCppDataType(o.referenceFile, "defaultBaseHeight", defaultBaseHeight);
  loadData::loadEigenMatrix(o.referenceFile, "defaultJointState", defaultMpcJointState);

  vector_t defaultFullJointState = vector_t::Zero(modelSettings.full_joint_dim);
  defaultFullJointState = mpcRobotModel.getFullModelJointAngles(defaultMpcJointState, defaultFullJointState);
  std::vector<std::string> jointNames(modelSettings.fullJointNames.begin(), modelSettings.fullJointNames.end());
  const std::vector<std::string> contactNames(modelSettings.contactNames.begin(), modelSettings.contactNames.end());
  FullModelProbe probe(o.urdfFile);
  // Contact frames are added to the MPC model only; measure them there.
  PinocchioInterface contactPinocchio = interface.getPinocchioInterface();
  std::vector<pinocchio::FrameIndex> contactFrames;
  for (const auto& name : contactNames) contactFrames.push_back(contactPinocchio.getModel().getFrameId(name));

  // Standing only: stance gait over the whole calibration, zero velocity, nominal upper body.
  const auto gaitMap = getGaitMap(o.gaitFile);
  if (gaitMap.find("stance") == gaitMap.end()) throw std::runtime_error("Gait `stance` missing from " + o.gaitFile);
  auto gaitSchedulePtr = interface.getSwitchedModelReferenceManagerPtr()->getGaitSchedule();
  GaitScheduleUpdater::updateGaitSchedule(gaitSchedulePtr, gaitMap.at("stance"), 0.0, interface.mpcSettings().timeHorizon_);

  CentroidalMpcTargetTrajectoriesCalculator targetCalculator(o.referenceFile, mpcRobotModel, interface.getPinocchioInterface(),
                                                             interface.getCentroidalModelInfo(), interface.mpcSettings().timeHorizon_);
  targetCalculator.setVelocityReferenceSmoothingEnabled(false);
  targetCalculator.setTargetJointState(defaultMpcJointState);
  const scalar_t waistYaw = defaultMpcJointState[mpcRobotModel.getJointIndex("waist_yaw_joint")];
  const scalar_t waistRoll = defaultMpcJointState[mpcRobotModel.getJointIndex("waist_roll_joint")];
  const scalar_t waistPitch = defaultMpcJointState[mpcRobotModel.getJointIndex("waist_pitch_joint")];

  SystemObservation observation;
  observation.time = 0.0;
  observation.state = interface.getInitialState();
  observation.input = vector_t::Zero(interface.getCentroidalModelInfo().inputDim);
  observation.mode = ModeNumber::STANCE;

  auto measureState = [&](const vector_t& state) {
    const vector_t fullJointPos = mpcRobotModel.getFullModelJointAngles(mpcRobotModel.getJointAngles(state), defaultFullJointState);
    Posture posture = probe.measure(mpcRobotModel.getBasePose(state), fullJointPos);
    const auto& model = contactPinocchio.getModel();
    auto& data = contactPinocchio.getData();
    pinocchio::forwardKinematics(model, data, mpcRobotModel.getGeneralizedCoordinates(state));
    pinocchio::updateFramePlacements(model, data);
    for (const auto id : contactFrames) posture.contactPos.push_back(data.oMf[id].translation());
    return posture;
  };

  const Posture initialPosture = measureState(observation.state);
  const scalar_t initialPelvis = initialPosture.basePose[2];
  const scalar_t initialTorso = initialPosture.torsoLinkPos.z();
  // Initial estimate: shift the pelvis by the torso height difference of the initial posture.
  scalar_t pelvisCommand = std::clamp(initialPelvis + (o.torsoOriginHeight - initialTorso), o.minPelvisHeight, o.maxPelvisHeight);
  std::cout << std::fixed << std::setprecision(5) << "[height_calibration] requested torso origin z=" << o.torsoOriginHeight
            << "  initial pelvis z=" << initialPelvis << "  initial torso origin z=" << initialTorso << "  first pelvis command=" << pelvisCommand
            << "\n";

  auto makeCommand = [&](scalar_t pelvisHeight) {
    WalkingVelocityCommand command;
    command.linear_velocity_x = 0.0;
    command.linear_velocity_y = 0.0;
    command.angular_velocity_z = 0.0;
    command.desired_pelvis_height = pelvisHeight;
    command.desired_waist_yaw = waistYaw;
    command.desired_waist_roll = waistRoll;
    command.desired_waist_pitch = waistPitch;
    return command;
  };

  {
    const auto command = makeCommand(defaultBaseHeight);
    for (int i = 0; i < 20; ++i) targetCalculator.commandedVelocityToTargetTrajectories(command, observation.time, observation.state);
  }
  mrt.resetMpcNode(TargetTrajectories({0.0}, {observation.state}, {observation.input}));
  mrt.setCurrentObservation(observation);
  for (size_t attempts = 0; !mrt.initialPolicyReceived(); ++attempts) {
    if (attempts > 20) throw std::runtime_error("Failed to receive the initial MPC policy.");
    mrt.advanceMpc();
    mrt.updatePolicy();
  }

  const size_t windowSteps = static_cast<size_t>(std::round(o.settleWindow / o.dt));
  const size_t minSteps = static_cast<size_t>(std::round(o.minSettleTime / o.dt));
  const size_t maxSteps = static_cast<size_t>(std::round(o.maxSettleTime / o.dt));

  // Advance the standing MPC with a fixed pelvis command until motion is low over the settle window.
  auto settle = [&](scalar_t pelvisHeight) {
    SettleResult result;
    std::deque<scalar_t> heights, comSpeeds, jointSpeeds;
    const auto command = makeCommand(pelvisHeight);
    for (size_t step = 1; step <= maxSteps; ++step) {
      const scalar_t time = observation.time;
      // Keep the stance schedule ahead of the horizon.
      GaitScheduleUpdater::updateGaitSchedule(gaitSchedulePtr, gaitMap.at("stance"), time, time + interface.mpcSettings().timeHorizon_);
      mrt.getReferenceManager().setTargetTrajectories(targetCalculator.commandedVelocityToTargetTrajectories(command, time, observation.state));
      mrt.setCurrentObservation(observation);
      mrt.advanceMpc();
      mrt.updatePolicy();

      vector_t nextState, nextInput;
      size_t nextMode = observation.mode;
      mrt.rolloutPolicy(time, observation.state, o.dt, nextState, nextInput, nextMode);
      if (!nextState.allFinite() || !nextInput.allFinite()) throw std::runtime_error("Non-finite MPC rollout state.");
      observation.time = time + o.dt;
      observation.state = std::move(nextState);
      observation.input = std::move(nextInput);
      observation.mode = nextMode;

      const auto flags = interface.getSwitchedModelReferenceManagerPtr()->getContactFlags(observation.time);
      for (size_t i = 0; i < flags.size(); ++i) result.doubleSupport = result.doubleSupport && flags[i];

      heights.push_back(measureState(observation.state).torsoLinkPos.z());
      comSpeeds.push_back(observation.state.head<3>().norm());  // normalized linear momentum = CoM velocity
      jointSpeeds.push_back(mpcRobotModel.getJointVelocities(observation.state, observation.input).cwiseAbs().maxCoeff());
      if (heights.size() > windowSteps) {
        heights.pop_front();
        comSpeeds.pop_front();
        jointSpeeds.pop_front();
      }
      result.duration = static_cast<scalar_t>(step) * o.dt;
      if (heights.size() == windowSteps) {
        const auto [lo, hi] = std::minmax_element(heights.begin(), heights.end());
        result.heightSpan = *hi - *lo;
        result.maxComSpeed = *std::max_element(comSpeeds.begin(), comSpeeds.end());
        result.maxJointSpeed = *std::max_element(jointSpeeds.begin(), jointSpeeds.end());
        if (step >= minSteps && result.heightSpan < o.settleHeightSpan && result.maxComSpeed < o.settleComSpeed &&
            result.maxJointSpeed < o.settleJointSpeed) {
          result.settled = true;
          return result;
        }
      }
    }
    return result;
  };

  std::vector<IterationRecord> history;
  // Secant search on f(command) = measured torso origin z. The MPC trades the pelvis target against its
  // joint-posture cost, so the torso moves only a fraction of each command change; the slope is measured.
  scalar_t slope = 1.0;
  for (size_t iteration = 0; iteration < o.maxIterations; ++iteration) {
    const SettleResult settled = settle(pelvisCommand);
    const Posture posture = measureState(observation.state);
    const scalar_t error = o.torsoOriginHeight - posture.torsoLinkPos.z();
    if (!history.empty()) {
      const auto& previous = history.back();
      const scalar_t dCommand = pelvisCommand - previous.pelvisCommand;
      if (std::abs(dCommand) > 1e-6) {
        slope = std::clamp((posture.torsoLinkPos.z() - previous.torsoOriginZ) / dCommand, o.minSlope, o.maxSlope);
      }
    }
    history.push_back({iteration, pelvisCommand, posture.basePose[2], posture.torsoLinkPos.z(), error, slope, settled});
    std::cout << "[height_calibration] iter " << iteration << " pelvis cmd=" << pelvisCommand << " pelvis z=" << posture.basePose[2]
              << " torso origin z=" << posture.torsoLinkPos.z() << " error=" << std::showpos << error << std::noshowpos
              << " slope=" << slope << " settled=" << settled.settled << " (" << settled.duration << " s)\n";
    if (!settled.doubleSupport) {
      writeResult(o, probe, jointNames, contactNames, false, "lost_double_support", history, &posture, pelvisCommand, initialPelvis, initialTorso);
      std::cerr << "[height_calibration] FAILED: contact schedule left double support.\n";
      return 2;
    }
    if (settled.settled && std::abs(error) < o.tolerance) {
      writeResult(o, probe, jointNames, contactNames, true, "converged", history, &posture, pelvisCommand, initialPelvis, initialTorso);
      std::cout << "[height_calibration] converged: whole-body CoM z=" << posture.wholeBodyCom.z() << " (torso origin z="
                << posture.torsoLinkPos.z() << ", pelvis z=" << posture.basePose[2] << ")\n[height_calibration] saved " << o.output << "\n";
      return 0;
    }
    const scalar_t next =
        std::clamp(pelvisCommand + std::clamp(o.gain * error / slope, -o.maxCorrection, o.maxCorrection), o.minPelvisHeight, o.maxPelvisHeight);
    // At a command bound with error still pushing outward, the height is unreachable.
    if (settled.settled && std::abs(next - pelvisCommand) < 1e-9) {
      writeResult(o, probe, jointNames, contactNames, false, "unreachable_pelvis_bound", history, &posture, pelvisCommand, initialPelvis,
                  initialTorso);
      std::cerr << "[height_calibration] FAILED: pelvis command saturated at " << pelvisCommand << " with torso error " << error << "\n";
      return 2;
    }
    pelvisCommand = next;
  }
  const Posture posture = measureState(observation.state);
  writeResult(o, probe, jointNames, contactNames, false, "max_iterations", history, &posture, pelvisCommand, initialPelvis, initialTorso);
  std::cerr << "[height_calibration] FAILED: no match within " << o.maxIterations << " iterations.\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parseOptions(argc, argv));
  } catch (const std::exception& e) {
    std::cerr << "[height_calibration] error: " << e.what() << "\n";
    return 1;
  }
}
