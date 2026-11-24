/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <humanoid_centroidal_mpc_ros2/MotionConditioning.h>
#include <humanoid_centroidal_mpc_ros2/FrameHeightCost.h>
#include <humanoid_centroidal_mpc_ros2/StepConditioning.h>

#include <sys/wait.h>
#include <unistd.h>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_core/cost/StateCost.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <ocs2_sqp/SqpMpc.h>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include <humanoid_common_mpc/gait/GaitScheduleUpdater.h>
#include <humanoid_common_mpc/gait/ModeSequenceTemplate.h>
#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/pinocchio_model/createPinocchioModel.h>
#include <humanoid_common_mpc/reference_manager/BreakFrequencyAlphaFilter.h>
#include <humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h>
#include <humanoid_common_mpc/swing_foot_planner/LipReferencePlan.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <zlib.h>
#include <Eigen/Geometry>

using namespace ocs2;
using namespace ocs2::humanoid;

namespace {

constexpr scalar_t kPi = 3.141592653589793238462643383279502884;

constexpr std::array<const char*, 30> kClampBodyNames = {
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
};

struct Options {
  std::string taskFile;
  std::string referenceFile;
  std::string urdfFile;
  std::string gaitFile;
  std::string output = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/generated_motions/g1_random_mpc_reference.npz";
  // Empty pairs preserve the original random command/gait workflow.
  std::vector<MotionCondition> conditions;
  std::vector<StepCommand> stepConditions;
  scalar_t doubleSupportFraction = 0.2 / 0.85;
  scalar_t footPositionWeight = 1000.0;
  scalar_t plateauRatio = 0.8;
  bool noStepTransitions = false;
  std::string conditionedGait = "slow_walk";
  bool checkConditioning = false;
  scalar_t duration = 20.0;
  scalar_t fps = 30.0;
  uint32_t seed = 1;
  size_t numMotions = 1;
  size_t maxAttemptsPerMotion = 20;
  scalar_t vxMin = -0.5;
  scalar_t vxMax = 1.0;
  scalar_t vyMin = -0.5;
  scalar_t vyMax = 0.5;
  scalar_t yawRateMin = -0.5;
  scalar_t yawRateMax = 0.5;
  scalar_t heightMin = 0.4;
  scalar_t heightMax = 0.8;
  scalar_t baseCommandDeadband = 0.1;
  scalar_t cmdVelSegmentMin = 3.0;
  scalar_t cmdVelSegmentMax = 9.0;
  scalar_t waistSegmentMin = 1.5;
  scalar_t waistSegmentMax = 4.5;
  scalar_t armSegmentMin = 2.0;
  scalar_t armSegmentMax = 6.0;
  scalar_t wristSegmentMin = 1.5;
  scalar_t wristSegmentMax = 4.5;
  scalar_t armLimitMargin = 0.05;
  scalar_t stanceProbability = 0.2;
  scalar_t headingProbability = 0.8;
  scalar_t headingMin = -kPi;
  scalar_t headingMax = kPi;
  scalar_t headingControlStiffness = 1.0;
  scalar_t upperBodyFixedProbability = 0.2;
  std::string upperBodyMode = "auto";  // Step-conditioned walking defaults to neutral targets.
  // LIP reference mode (LIP/scripts/export_lip_to_mpc.py CSV). Exclusive with step/velocity command modes.
  std::string lipReference;
  bool durationGiven = false;
  // LIP mode: torso_link origin height tracked over every MPC horizon (default target: the CSV's requested height).
  std::optional<scalar_t> torsoOriginHeight;
  scalar_t torsoHeightWeight = 10000.0;
  std::vector<std::string> commandModeOptions;  // command-mode options seen on the command line
};

// CoM/foot samples exported from LIP (format lip_reference_csv_v1). Feet are MPC foot_*_contact frames.
struct LipReference {
  std::unordered_map<std::string, std::string> metadata;
  scalar_t dt = 0.0;
  scalar_t heading = 0.0;
  // Calibrated posture the LIP reference was generated from (contact frames at z=0).
  vector6_t initialBasePose = vector6_t::Zero();  // x, y, z, yaw, pitch, roll
  std::unordered_map<std::string, scalar_t> initialJointPos;
  std::vector<scalar_t> time;
  std::vector<std::array<bool, 2>> contact;  // left, right; true = stance
  std::vector<vector3_t> comPos, comVel;
  std::vector<std::array<vector3_t, 2>> footPos, footVel;
  size_t size() const { return time.size(); }
  scalar_t duration() const { return time.back(); }
};


struct SmoothScalarTrajectory {
  std::vector<scalar_t> times;
  std::vector<scalar_t> values;

  scalar_t value(scalar_t time) const {
    if (times.empty() || values.empty()) {
      throw std::runtime_error("SmoothScalarTrajectory is empty.");
    }
    if (time <= times.front()) {
      return values.front();
    }
    if (time >= times.back()) {
      return values.back();
    }

    const auto upper = std::upper_bound(times.begin(), times.end(), time);
    const size_t i1 = static_cast<size_t>(std::distance(times.begin(), upper));
    const size_t i0 = i1 - 1;
    const scalar_t alpha = (time - times[i0]) / (times[i1] - times[i0]);
    const scalar_t smoothAlpha = 0.5 - 0.5 * std::cos(kPi * alpha);
    return values[i0] + smoothAlpha * (values[i1] - values[i0]);
  }
};

struct StanceSchedule {
  std::vector<scalar_t> times;
  std::vector<bool> values;

  bool value(scalar_t time) const {
    if (times.empty()) {
      return false;
    }
    const auto upper = std::upper_bound(times.begin(), times.end(), time);
    if (upper == times.begin()) {
      return values.front();
    }
    return values[static_cast<size_t>(std::distance(times.begin(), upper) - 1)];
  }
};

struct HeadingSchedule {
  std::vector<scalar_t> times;
  std::vector<bool> active;
  std::vector<scalar_t> targets;

  std::pair<bool, scalar_t> value(scalar_t time) const {
    if (times.empty()) {
      return {false, 0.0};
    }
    const auto upper = std::upper_bound(times.begin(), times.end(), time);
    size_t index = 0;
    if (upper != times.begin()) {
      index = static_cast<size_t>(std::distance(times.begin(), upper) - 1);
    }
    return {active[index], targets[index]};
  }
};

struct MotionBuffers {
  std::vector<float> jointPos;
  std::vector<float> jointVel;
  std::vector<float> bodyPos;
  std::vector<float> bodyQuat;
  std::vector<float> bodyLinVel;
  std::vector<float> bodyAngVel;
  std::vector<float> contactWrench;
  std::vector<uint8_t> contactFlags;
  std::vector<float> stepConditioning, stepConditioningRequested, stepConditioningTarget;
  std::vector<float> footReferencePos, footReferenceVel;
  std::vector<double> plannedSteps;
  // LIP mode: reference sampled at each exported frame time, and the measured whole-body CoM.
  std::vector<float> lipComReference, lipComVelReference, lipFootReference, lipFootVelReference, comPos;
  std::vector<uint8_t> lipContact;
  std::vector<float> conditioning;
  std::vector<float> conditioningTarget;
  std::vector<float> commandedVelocity;
  std::vector<uint8_t> transition;
  std::vector<int32_t> conditionStage;
  std::vector<double> commandTime;
  std::vector<double> sampleTime;
  std::vector<double> fps;
  size_t numFrames = 0;
  size_t numJoints = 0;
  size_t numBodies = 0;
  size_t numContacts = 0;
};

struct ZipEntry {
  std::string filename;
  std::vector<uint8_t> payload;
  std::vector<uint8_t> compressedPayload;
  uint32_t crc = 0;
  uint32_t offset = 0;
};

uint32_t crc32(const uint8_t* data, size_t length) {
  uint32_t crc = 0xffffffffu;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t mask = -(crc & 1u);
      crc = (crc >> 1u) ^ (0xedb88320u & mask);
    }
  }
  return ~crc;
}

std::vector<uint8_t> deflateRaw(const std::vector<uint8_t>& input) {
  if (input.empty()) {
    return {};
  }

  z_stream stream{};
  const int initResult = deflateInit2(&stream, Z_BEST_SPEED, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY);
  if (initResult != Z_OK) {
    throw std::runtime_error("Failed to initialize zlib deflate stream.");
  }

  std::vector<uint8_t> output(deflateBound(&stream, static_cast<uLong>(input.size())));
  stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
  stream.avail_in = static_cast<uInt>(input.size());
  stream.next_out = reinterpret_cast<Bytef*>(output.data());
  stream.avail_out = static_cast<uInt>(output.size());

  const int deflateResult = deflate(&stream, Z_FINISH);
  if (deflateResult != Z_STREAM_END) {
    deflateEnd(&stream);
    throw std::runtime_error("Failed to deflate NPZ entry.");
  }

  output.resize(stream.total_out);
  deflateEnd(&stream);
  return output;
}

void appendU16(std::vector<uint8_t>& out, uint16_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xffu));
  out.push_back(static_cast<uint8_t>((value >> 8u) & 0xffu));
}

void appendU32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xffu));
  out.push_back(static_cast<uint8_t>((value >> 8u) & 0xffu));
  out.push_back(static_cast<uint8_t>((value >> 16u) & 0xffu));
  out.push_back(static_cast<uint8_t>((value >> 24u) & 0xffu));
}

std::string shapeString(const std::vector<size_t>& shape) {
  std::ostringstream ss;
  ss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << shape[i];
  }
  if (shape.size() == 1) {
    ss << ",";
  }
  ss << ")";
  return ss.str();
}

std::vector<uint8_t> makeNpyHeader(const std::string& descr, const std::vector<size_t>& shape) {
  std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shapeString(shape) + ", }";
  const size_t prefixLength = 10;
  const size_t padding = 16 - ((prefixLength + header.size() + 1) % 16);
  header.append(padding, ' ');
  header.push_back('\n');

  if (header.size() > std::numeric_limits<uint16_t>::max()) {
    throw std::runtime_error("NPY header is too large for v1.0 format.");
  }

  std::vector<uint8_t> out;
  out.reserve(prefixLength + header.size());
  out.push_back(0x93);
  const std::string magic = "NUMPY";
  out.insert(out.end(), magic.begin(), magic.end());
  out.push_back(1);
  out.push_back(0);
  appendU16(out, static_cast<uint16_t>(header.size()));
  out.insert(out.end(), header.begin(), header.end());
  return out;
}

template <typename T>
std::vector<uint8_t> makeNumericNpy(const std::vector<T>& data, const std::string& descr, const std::vector<size_t>& shape) {
  std::vector<uint8_t> out = makeNpyHeader(descr, shape);
  const auto* bytes = reinterpret_cast<const uint8_t*>(data.data());
  out.insert(out.end(), bytes, bytes + data.size() * sizeof(T));
  return out;
}

std::vector<uint8_t> makeStringNpy(const std::vector<std::string>& strings, size_t width) {
  std::vector<uint8_t> out = makeNpyHeader("|S" + std::to_string(width), {strings.size()});
  const size_t payloadOffset = out.size();
  out.resize(payloadOffset + strings.size() * width, 0);
  for (size_t i = 0; i < strings.size(); ++i) {
    const size_t n = std::min(width, strings[i].size());
    std::copy_n(strings[i].data(), n, out.data() + payloadOffset + i * width);
  }
  return out;
}

void writeZip(const std::filesystem::path& path, std::vector<ZipEntry> entries) {
  std::vector<uint8_t> out;
  for (auto& entry : entries) {
    entry.crc = crc32(entry.payload.data(), entry.payload.size());
    entry.compressedPayload = deflateRaw(entry.payload);
    entry.offset = static_cast<uint32_t>(out.size());

    appendU32(out, 0x04034b50u);
    appendU16(out, 20);
    appendU16(out, 0);
    appendU16(out, 8);
    appendU16(out, 0);
    appendU16(out, 0);
    appendU32(out, entry.crc);
    appendU32(out, static_cast<uint32_t>(entry.compressedPayload.size()));
    appendU32(out, static_cast<uint32_t>(entry.payload.size()));
    appendU16(out, static_cast<uint16_t>(entry.filename.size()));
    appendU16(out, 0);
    out.insert(out.end(), entry.filename.begin(), entry.filename.end());
    out.insert(out.end(), entry.compressedPayload.begin(), entry.compressedPayload.end());
  }

  const uint32_t centralDirectoryOffset = static_cast<uint32_t>(out.size());
  for (const auto& entry : entries) {
    appendU32(out, 0x02014b50u);
    appendU16(out, 20);
    appendU16(out, 20);
    appendU16(out, 0);
    appendU16(out, 8);
    appendU16(out, 0);
    appendU16(out, 0);
    appendU32(out, entry.crc);
    appendU32(out, static_cast<uint32_t>(entry.compressedPayload.size()));
    appendU32(out, static_cast<uint32_t>(entry.payload.size()));
    appendU16(out, static_cast<uint16_t>(entry.filename.size()));
    appendU16(out, 0);
    appendU16(out, 0);
    appendU16(out, 0);
    appendU16(out, 0);
    appendU32(out, 0);
    appendU32(out, entry.offset);
    out.insert(out.end(), entry.filename.begin(), entry.filename.end());
  }
  const uint32_t centralDirectorySize = static_cast<uint32_t>(out.size()) - centralDirectoryOffset;

  appendU32(out, 0x06054b50u);
  appendU16(out, 0);
  appendU16(out, 0);
  appendU16(out, static_cast<uint16_t>(entries.size()));
  appendU16(out, static_cast<uint16_t>(entries.size()));
  appendU32(out, centralDirectorySize);
  appendU32(out, centralDirectoryOffset);
  appendU16(out, 0);

  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open output NPZ path: " + path.string());
  }
  file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
}

void saveClampNpz(const std::filesystem::path& path,
                  const MotionBuffers& buffers,
                  const std::vector<std::string>& jointNames,
                  const std::vector<std::string>& bodyNames,
                  const std::vector<std::string>& contactNames) {
  const size_t expectedJointScalars = buffers.numFrames * buffers.numJoints;
  const size_t expectedBodyPositionScalars = buffers.numFrames * buffers.numBodies * 3;
  const size_t expectedBodyQuaternionScalars = buffers.numFrames * buffers.numBodies * 4;
  const size_t expectedContactWrenchScalars = buffers.numFrames * buffers.numContacts * 6;
  const size_t expectedContactFlagScalars = buffers.numFrames * buffers.numContacts;
  if (buffers.jointPos.size() != expectedJointScalars || buffers.jointVel.size() != expectedJointScalars ||
      buffers.bodyPos.size() != expectedBodyPositionScalars || buffers.bodyLinVel.size() != expectedBodyPositionScalars ||
      buffers.bodyAngVel.size() != expectedBodyPositionScalars || buffers.bodyQuat.size() != expectedBodyQuaternionScalars ||
      buffers.contactWrench.size() != expectedContactWrenchScalars || buffers.contactFlags.size() != expectedContactFlagScalars) {
    throw std::runtime_error("Motion buffer sizes do not match the requested NPZ shapes.");
  }

  std::vector<ZipEntry> entries;
  entries.push_back({"joint_pos.npy", makeNumericNpy(buffers.jointPos, "<f4", {buffers.numFrames, buffers.numJoints})});
  entries.push_back({"joint_vel.npy", makeNumericNpy(buffers.jointVel, "<f4", {buffers.numFrames, buffers.numJoints})});
  entries.push_back({"body_pos_w.npy", makeNumericNpy(buffers.bodyPos, "<f4", {buffers.numFrames, buffers.numBodies, 3})});
  entries.push_back({"body_quat_w.npy", makeNumericNpy(buffers.bodyQuat, "<f4", {buffers.numFrames, buffers.numBodies, 4})});
  entries.push_back({"body_lin_vel_w.npy", makeNumericNpy(buffers.bodyLinVel, "<f4", {buffers.numFrames, buffers.numBodies, 3})});
  entries.push_back({"body_ang_vel_w.npy", makeNumericNpy(buffers.bodyAngVel, "<f4", {buffers.numFrames, buffers.numBodies, 3})});
  entries.push_back({"contact_wrench_w.npy", makeNumericNpy(buffers.contactWrench, "<f4", {buffers.numFrames, buffers.numContacts, 6})});
  entries.push_back({"contact_flags.npy", makeNumericNpy(buffers.contactFlags, "|u1", {buffers.numFrames, buffers.numContacts})});
  entries.push_back({"body_names.npy", makeStringNpy(bodyNames, 32)});
  entries.push_back({"body_link_names.npy", makeStringNpy(bodyNames, 32)});
  entries.push_back({"contact_names.npy", makeStringNpy(contactNames, 32)});
  entries.push_back({"joint_names.npy", makeStringNpy(jointNames, 40)});
  entries.push_back({"fps.npy", makeNumericNpy(buffers.fps, "<f8", {1})});
  if (!buffers.stepConditioning.empty()) {
    if (buffers.stepConditioning.size() != buffers.numFrames * 2 ||
        buffers.stepConditioningRequested.size() != buffers.numFrames * 2 ||
        buffers.stepConditioningTarget.size() != buffers.numFrames * 2 ||
        buffers.footReferencePos.size() != buffers.numFrames * 4 ||
        buffers.footReferenceVel.size() != buffers.numFrames * 4 || buffers.plannedSteps.size() % 8 != 0)
      throw std::runtime_error("Step reference buffers do not match exported frames.");
    entries.push_back({"step_conditioning.npy", makeNumericNpy(buffers.stepConditioning, "<f4", {buffers.numFrames, 2})});
    entries.push_back({"step_conditioning_requested.npy", makeNumericNpy(buffers.stepConditioningRequested, "<f4", {buffers.numFrames, 2})});
    entries.push_back({"step_conditioning_target.npy", makeNumericNpy(buffers.stepConditioningTarget, "<f4", {buffers.numFrames, 2})});
    entries.push_back({"step_conditioning_names.npy", makeStringNpy({"step_length_m", "step_duration_s"}, 32)});
    entries.push_back({"foot_reference_pos_xy_w.npy", makeNumericNpy(buffers.footReferencePos, "<f4", {buffers.numFrames, 2, 2})});
    entries.push_back({"foot_reference_vel_xy_w.npy", makeNumericNpy(buffers.footReferenceVel, "<f4", {buffers.numFrames, 2, 2})});
    entries.push_back({"foot_reference_names.npy", makeStringNpy({"left", "right"}, 16)});
    entries.push_back({"planned_steps.npy", makeNumericNpy(buffers.plannedSteps, "<f8", {buffers.plannedSteps.size() / 8, 8})});
    entries.push_back({"planned_step_columns.npy", makeStringNpy({"start_s", "liftoff_s", "touchdown_s", "foot_index", "step_length_m", "step_duration_s", "touchdown_x_w", "touchdown_y_w"}, 32)});
  }
  if (!buffers.lipComReference.empty()) {
    const size_t n = buffers.numFrames;
    if (buffers.lipComReference.size() != n * 3 || buffers.lipComVelReference.size() != n * 3 || buffers.comPos.size() != n * 3 ||
        buffers.lipFootReference.size() != n * 6 || buffers.lipFootVelReference.size() != n * 6 || buffers.lipContact.size() != n * 2)
      throw std::runtime_error("LIP reference buffers do not match exported frames.");
    entries.push_back({"lip_com_pos_w.npy", makeNumericNpy(buffers.lipComReference, "<f4", {n, 3})});
    entries.push_back({"lip_com_vel_w.npy", makeNumericNpy(buffers.lipComVelReference, "<f4", {n, 3})});
    entries.push_back({"lip_foot_pos_w.npy", makeNumericNpy(buffers.lipFootReference, "<f4", {n, 2, 3})});
    entries.push_back({"lip_foot_vel_w.npy", makeNumericNpy(buffers.lipFootVelReference, "<f4", {n, 2, 3})});
    entries.push_back({"lip_contact.npy", makeNumericNpy(buffers.lipContact, "|u1", {n, 2})});
    entries.push_back({"lip_foot_names.npy", makeStringNpy({"left", "right"}, 16)});
    entries.push_back({"com_pos_w.npy", makeNumericNpy(buffers.comPos, "<f4", {n, 3})});
  }
  if (!buffers.conditioning.empty()) {
    if (buffers.conditioning.size() != buffers.numFrames * 2 || buffers.conditioningTarget.size() != buffers.numFrames * 2 ||
        buffers.commandedVelocity.size() != buffers.numFrames * 4 || buffers.transition.size() != buffers.numFrames ||
        buffers.conditionStage.size() != buffers.numFrames || buffers.commandTime.size() != buffers.numFrames ||
        buffers.sampleTime.size() != buffers.numFrames) {
      throw std::runtime_error("Conditioning buffer sizes do not match motion frames.");
    }
    entries.push_back({"conditioning.npy", makeNumericNpy(buffers.conditioning, "<f4", {buffers.numFrames, 2})});
    entries.push_back({"conditioning_target.npy", makeNumericNpy(buffers.conditioningTarget, "<f4", {buffers.numFrames, 2})});
    entries.push_back({"conditioning_names.npy", makeStringNpy({"forward_velocity_mps", "full_stride_length_m"}, 32)});
    entries.push_back({"commanded_velocity.npy", makeNumericNpy(buffers.commandedVelocity, "<f4", {buffers.numFrames, 4})});
    entries.push_back({"commanded_velocity_names.npy", makeStringNpy({"vx_body_mps", "vy_body_mps", "base_height_m", "yaw_rate_rps"}, 32)});
    entries.push_back({"is_transition.npy", makeNumericNpy(buffers.transition, "|u1", {buffers.numFrames})});
    entries.push_back({"condition_stage.npy", makeNumericNpy(buffers.conditionStage, "<i4", {buffers.numFrames})});
    entries.push_back({"command_time.npy", makeNumericNpy(buffers.commandTime, "<f8", {buffers.numFrames})});
    entries.push_back({"sample_time.npy", makeNumericNpy(buffers.sampleTime, "<f8", {buffers.numFrames})});
  }
  writeZip(path, std::move(entries));
}

std::string requireValue(int& index, int argc, char** argv) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("Missing value for option ") + argv[index]);
  }
  ++index;
  return argv[index];
}

scalar_t parseScalar(const std::string& text, const std::string& option) {
  try {
    return std::stod(text);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid scalar value for " + option + ": " + text);
  }
}

void requireOrderedRange(scalar_t lower, scalar_t upper, const std::string& name) {
  if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper) {
    throw std::runtime_error(name + " must be finite with min <= max.");
  }
}

void requirePositiveSegmentRange(scalar_t lower, scalar_t upper, const std::string& name) {
  if (!std::isfinite(lower) || !std::isfinite(upper) || lower <= 0.0 || upper <= 0.0 || lower > upper) {
    throw std::runtime_error(name + " must be finite, positive, and min <= max.");
  }
}

std::string trim(const std::string& text) {
  const auto begin = text.find_first_not_of(" \t\r");
  if (begin == std::string::npos) return "";
  return text.substr(begin, text.find_last_not_of(" \t\r") - begin + 1);
}

std::vector<std::string> splitComma(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(trim(field));
  if (!line.empty() && line.back() == ',') fields.emplace_back();
  return fields;
}

LipReference loadLipReference(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Cannot open LIP reference: " + path);
  const std::vector<std::string> expectedColumns = {
      "time", "left_contact", "right_contact", "com_x", "com_y", "com_z", "com_vx", "com_vy", "com_vz",
      "left_x", "left_y", "left_z", "right_x", "right_y", "right_z",
      "left_vx", "left_vy", "left_vz", "right_vx", "right_vy", "right_vz"};
  const auto fail = [&](const std::string& message, size_t line = 0) {
    throw std::runtime_error("LIP reference " + path + (line ? " line " + std::to_string(line) : std::string()) + ": " + message);
  };

  LipReference ref;
  std::string line;
  size_t lineNumber = 0;
  bool headerSeen = false;
  while (std::getline(file, line)) {
    ++lineNumber;
    if (trim(line).empty()) continue;
    if (line[0] == '#') {
      if (headerSeen) fail("metadata after the column header", lineNumber);
      const auto colon = line.find(':');
      if (colon == std::string::npos) fail("metadata line without 'key: value'", lineNumber);
      ref.metadata[trim(line.substr(1, colon - 1))] = trim(line.substr(colon + 1));
      continue;
    }
    const auto fields = splitComma(line);
    if (!headerSeen) {
      if (fields != expectedColumns) fail("unexpected column header", lineNumber);
      headerSeen = true;
      continue;
    }
    if (fields.size() != expectedColumns.size()) fail("expected " + std::to_string(expectedColumns.size()) + " fields", lineNumber);
    std::vector<scalar_t> v(fields.size());
    for (size_t i = 0; i < fields.size(); ++i) {
      size_t used = 0;
      try {
        v[i] = std::stod(fields[i], &used);
      } catch (const std::exception&) {
        used = 0;
      }
      if (used != fields[i].size() || !std::isfinite(v[i])) fail("invalid number '" + fields[i] + "' in " + expectedColumns[i], lineNumber);
    }
    for (size_t i : {1, 2})
      if (v[i] != 0.0 && v[i] != 1.0) fail(expectedColumns[i] + " must be 0 or 1", lineNumber);
    ref.time.push_back(v[0]);
    ref.contact.push_back({v[1] == 1.0, v[2] == 1.0});
    ref.comPos.emplace_back(v[3], v[4], v[5]);
    ref.comVel.emplace_back(v[6], v[7], v[8]);
    ref.footPos.push_back({vector3_t(v[9], v[10], v[11]), vector3_t(v[12], v[13], v[14])});
    ref.footVel.push_back({vector3_t(v[15], v[16], v[17]), vector3_t(v[18], v[19], v[20])});
  }
  if (!headerSeen) fail("missing column header");

  const auto meta = [&](const std::string& key) {
    const auto it = ref.metadata.find(key);
    if (it == ref.metadata.end()) fail("missing metadata '" + key + "'");
    return it->second;
  };
  if (meta("format") != "lip_reference_csv_v2") fail("format must be lip_reference_csv_v2, got " + meta("format") + " (re-export)");
  if (meta("frame") != "world_z_up") fail("frame must be world_z_up, got " + meta("frame"));
  ref.dt = parseScalar(meta("dt"), "dt metadata");
  const scalar_t lipComHeight = parseScalar(meta("lip_com_height"), "lip_com_height metadata");
  ref.heading = parseScalar(meta("heading"), "heading metadata");
  const auto basePose = splitComma(meta("initial_base_pose_xyz_yaw_pitch_roll"));
  if (basePose.size() != 6) fail("initial_base_pose_xyz_yaw_pitch_roll needs 6 values");
  for (size_t i = 0; i < 6; ++i) ref.initialBasePose[i] = parseScalar(basePose[i], "initial_base_pose_xyz_yaw_pitch_roll");
  const auto jointNames = splitComma(meta("initial_joint_names"));
  const auto jointPos = splitComma(meta("initial_joint_pos"));
  if (jointNames.empty() || jointNames.size() != jointPos.size()) fail("initial_joint_names/initial_joint_pos lengths differ");
  for (size_t i = 0; i < jointNames.size(); ++i) {
    const scalar_t q = parseScalar(jointPos[i], "initial_joint_pos");
    if (!std::isfinite(q) || !ref.initialJointPos.emplace(jointNames[i], q).second) fail("bad or duplicate joint " + jointNames[i]);
  }
  if (!ref.initialBasePose.allFinite()) fail("non-finite initial base pose");
  if (!(ref.dt > 0.0)) fail("dt must be positive");
  if (ref.size() < 2) fail("needs at least two samples");
  if (std::to_string(ref.size()) != meta("samples")) fail("samples metadata " + meta("samples") + " != rows " + std::to_string(ref.size()));
  if (std::abs(ref.time.front()) > 1e-9) fail("time must start at 0");
  for (size_t k = 1; k < ref.size(); ++k) {
    const scalar_t step = ref.time[k] - ref.time[k - 1];
    // The exporter keeps the final sample, so only the last interval may be shorter.
    const bool ok = k + 1 < ref.size() ? std::abs(step - ref.dt) < 1e-6 : step > 0.0 && step < ref.dt + 1e-6;
    if (!ok) fail("non-uniform time step " + std::to_string(step) + " at t=" + std::to_string(ref.time[k]));
  }
  if (std::abs(ref.duration() - parseScalar(meta("duration"), "duration metadata")) > 1e-6) fail("duration metadata does not match time");
  if (!ref.contact.front()[0] || !ref.contact.front()[1]) fail("must start in double support");
  for (size_t k = 0; k < ref.size(); ++k) {
    if (!ref.contact[k][0] && !ref.contact[k][1]) fail("flight phase at t=" + std::to_string(ref.time[k]));
    for (size_t foot = 0; foot < 2; ++foot)
      if ((ref.contact[k][foot] && std::abs(ref.footPos[k][foot].z()) > 1e-6) || ref.footPos[k][foot].z() < -1e-6)
        fail("stance foot off the ground or foot below it at t=" + std::to_string(ref.time[k]));
    if (std::abs(ref.comPos[k].z() - lipComHeight) > 1e-6) fail("CoM z differs from lip_com_height at t=" + std::to_string(ref.time[k]));
  }
  const auto calibration = meta("height_calibration");
  if (!std::filesystem::is_regular_file(calibration)) fail("height_calibration " + calibration + " is not visible here");
  return ref;
}

Options parseOptions(int argc, char** argv) {
  Options options;

  const auto g1MpcShare = ament_index_cpp::get_package_share_directory("g1_centroidal_mpc");
  const auto g1DescriptionShare = ament_index_cpp::get_package_share_directory("g1_description");
  const auto commonMpcShare = ament_index_cpp::get_package_share_directory("humanoid_common_mpc");
  options.taskFile = g1MpcShare + "/config/mpc/task_random_reference.info";
  options.referenceFile = g1MpcShare + "/config/command/reference_random_reference.info";
  options.urdfFile = g1DescriptionShare + "/urdf/g1_29dof.urdf";
  options.gaitFile = commonMpcShare + "/config/command/gait.info";

  static const std::vector<std::string> kCommandModeOptions = {
      "--step-condition", "--condition", "--no-step-transitions", "--plateau-ratio", "--conditioned-gait",
      "--check-conditioning", "--double-support-fraction",
      "--vx-min", "--vx-max", "--vy-min", "--vy-max", "--yaw-rate-min", "--yaw-rate-max", "--height-min", "--height-max",
      "--base-command-deadband", "--cmd-vel-segment-min", "--cmd-vel-segment-max", "--stance-probability",
      "--heading-prob", "--heading-probability", "--heading_prob", "--heading-min", "--heading-max",
      "--heading-control-stiffness"};
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (std::find(kCommandModeOptions.begin(), kCommandModeOptions.end(), arg) != kCommandModeOptions.end()) {
      options.commandModeOptions.push_back(arg);
    }
    if (arg == "--lip-reference") {
      options.lipReference = requireValue(i, argc, argv);
    } else if (arg == "--torso-origin-height") {
      options.torsoOriginHeight = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--torso-height-weight") {
      options.torsoHeightWeight = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--step-condition") {
      const auto value = requireValue(i, argc, argv);
      const auto comma = value.find(',');
      if (comma == std::string::npos || value.find(',', comma + 1) != std::string::npos)
        throw std::runtime_error("--step-condition expects STEP_LENGTH_METERS,STEP_DURATION_SECONDS.");
      options.stepConditions.push_back({parseScalar(value.substr(0, comma), arg), parseScalar(value.substr(comma + 1), arg)});
    } else if (arg == "--double-support-fraction") {
      options.doubleSupportFraction = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--foot-position-weight") {
      options.footPositionWeight = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--condition") {
      const auto value = requireValue(i, argc, argv);
      const auto comma = value.find(',');
      if (comma == std::string::npos || value.find(',', comma + 1) != std::string::npos) {
        throw std::runtime_error("--condition expects VELOCITY_MPS,FULL_STRIDE_METERS (repeat for multiple stages).");
      }
      options.conditions.push_back({parseScalar(value.substr(0, comma), arg), parseScalar(value.substr(comma + 1), arg)});
    } else if (arg == "--no-step-transitions") {
      options.noStepTransitions = true;
    } else if (arg == "--plateau-ratio") {
      options.plateauRatio = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--conditioned-gait") {
      options.conditionedGait = requireValue(i, argc, argv);
    } else if (arg == "--check-conditioning") {
      options.checkConditioning = true;
    } else if (arg == "--task-file") {
      options.taskFile = requireValue(i, argc, argv);
    } else if (arg == "--reference-file") {
      options.referenceFile = requireValue(i, argc, argv);
    } else if (arg == "--urdf-file") {
      options.urdfFile = requireValue(i, argc, argv);
    } else if (arg == "--gait-file") {
      options.gaitFile = requireValue(i, argc, argv);
    } else if (arg == "--output") {
      options.output = requireValue(i, argc, argv);
    } else if (arg == "--duration") {
      options.duration = parseScalar(requireValue(i, argc, argv), arg);
      options.durationGiven = true;
    } else if (arg == "--fps") {
      options.fps = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--seed") {
      options.seed = static_cast<uint32_t>(std::stoul(requireValue(i, argc, argv)));
    } else if (arg == "--num-motions" || arg == "--motions") {
      options.numMotions = static_cast<size_t>(std::stoul(requireValue(i, argc, argv)));
    } else if (arg == "--max-attempts-per-motion") {
      options.maxAttemptsPerMotion = static_cast<size_t>(std::stoul(requireValue(i, argc, argv)));
    } else if (arg == "--vx-min") {
      options.vxMin = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--vx-max") {
      options.vxMax = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--vy-min") {
      options.vyMin = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--vy-max") {
      options.vyMax = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--yaw-rate-min") {
      options.yawRateMin = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--yaw-rate-max") {
      options.yawRateMax = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--height-min") {
      options.heightMin = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--height-max") {
      options.heightMax = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--base-command-deadband") {
      options.baseCommandDeadband = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--cmd-vel-segment-min") {
      options.cmdVelSegmentMin = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--cmd-vel-segment-max") {
      options.cmdVelSegmentMax = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--stance-probability") {
      options.stanceProbability = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--heading-prob" || arg == "--heading-probability" || arg == "--heading_prob") {
      options.headingProbability = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--heading-min") {
      options.headingMin = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--heading-max") {
      options.headingMax = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--heading-control-stiffness") {
      options.headingControlStiffness = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--upper-body-mode") {
      options.upperBodyMode = requireValue(i, argc, argv);
    } else if (arg == "--upper-body-fixed-probability") {
      options.upperBodyFixedProbability = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--wrist-segment-min") {
      options.wristSegmentMin = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--wrist-segment-max") {
      options.wristSegmentMax = parseScalar(requireValue(i, argc, argv), arg);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--output PATH] [--duration SEC] [--fps HZ] [--seed N] [--num-motions N]\n"
                << "       [--lip-reference CSV] (LIP CoM/foot samples from LIP/scripts/export_lip_to_mpc.py;\n"
                << "        excludes step, velocity-condition, and random command options; --foot-position-weight applies;\n"
                << "        default --duration is the reference duration minus the MPC horizon)\n"
                << "       [--torso-origin-height M] [--torso-height-weight W] (LIP mode: torso_link origin height cost over the\n"
                << "        whole horizon; default target is the CSV requested_torso_origin_height, default weight 10000, 0 disables)\n"
                << "       [--step-condition STEP_LENGTH_METERS,STEP_DURATION_SECONDS] (repeatable; explicit foot targets)\n"
                << "       [--double-support-fraction P] [--foot-position-weight W] [--no-step-transitions]\n"
                << "       [--condition VELOCITY_MPS,FULL_STRIDE_METERS] (repeatable, ordered pairs)\n"
                << "       [--plateau-ratio P] [--conditioned-gait NAME] [--check-conditioning]\n"
                << "       [--max-attempts-per-motion N]\n"
                << "       [--vx-min V] [--vx-max V] [--vy-min V] [--vy-max V]\n"
                << "       [--yaw-rate-min V] [--yaw-rate-max V] [--height-min H] [--height-max H]\n"
                << "       [--base-command-deadband V]\n"
                << "       [--cmd-vel-segment-min SEC] [--cmd-vel-segment-max SEC]\n"
                << "       [--upper-body-mode auto|neutral|random] (auto: neutral for step conditions)\n"
                << "       [--stance-probability P] [--heading-prob P] [--upper-body-fixed-probability P]\n"
                << "       [--heading-min RAD] [--heading-max RAD] [--heading-control-stiffness K]\n"
                << "       [--wrist-segment-min SEC] [--wrist-segment-max SEC]\n"
                << "       [--task-file PATH] [--reference-file PATH] [--urdf-file PATH] [--gait-file PATH]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  if (!options.lipReference.empty() && !options.commandModeOptions.empty()) {
    std::string seen;
    for (const auto& name : options.commandModeOptions) seen += (seen.empty() ? "" : ", ") + name;
    throw std::runtime_error("--lip-reference cannot be combined with step, velocity-condition, or random command options (got " +
                             seen + ").");
  }
  if (options.lipReference.empty() && (options.torsoOriginHeight || options.torsoHeightWeight != 10000.0)) {
    throw std::runtime_error("--torso-origin-height/--torso-height-weight require --lip-reference.");
  }
  if (!std::isfinite(options.torsoHeightWeight) || options.torsoHeightWeight < 0.0 ||
      (options.torsoOriginHeight && !(std::isfinite(*options.torsoOriginHeight) && *options.torsoOriginHeight > 0.0))) {
    throw std::runtime_error("--torso-height-weight must be finite and >= 0; --torso-origin-height finite and positive.");
  }
  if (!options.lipReference.empty() && options.numMotions != 1) {
    throw std::runtime_error("--lip-reference generates exactly one motion; drop --num-motions.");
  }
  if (options.duration <= 0.0) {
    throw std::runtime_error("--duration must be positive.");
  }
  if (options.fps <= 0.0) {
    throw std::runtime_error("--fps must be positive.");
  }
  if (options.numMotions == 0) {
    throw std::runtime_error("--num-motions must be positive.");
  }
  if (options.maxAttemptsPerMotion == 0) {
    throw std::runtime_error("--max-attempts-per-motion must be positive.");
  }
  if (options.upperBodyMode != "auto" && options.upperBodyMode != "neutral" && options.upperBodyMode != "random") {
    throw std::runtime_error("--upper-body-mode must be auto, neutral, or random.");
  }
  if (options.upperBodyFixedProbability < 0.0 || options.upperBodyFixedProbability > 1.0) {
    throw std::runtime_error("--upper-body-fixed-probability must be in [0, 1].");
  }
  if (options.stanceProbability < 0.0 || options.stanceProbability > 1.0) {
    throw std::runtime_error("--stance-probability must be in [0, 1].");
  }
  if (options.headingProbability < 0.0 || options.headingProbability > 1.0) {
    throw std::runtime_error("--heading-prob must be in [0, 1].");
  }
  if (options.headingControlStiffness < 0.0) {
    throw std::runtime_error("--heading-control-stiffness must be non-negative.");
  }
  if (options.baseCommandDeadband < 0.0) {
    throw std::runtime_error("--base-command-deadband must be non-negative.");
  }
  requireOrderedRange(options.vxMin, options.vxMax, "--vx-min/max");
  requireOrderedRange(options.vyMin, options.vyMax, "--vy-min/max");
  requireOrderedRange(options.yawRateMin, options.yawRateMax, "--yaw-rate-min/max");
  requireOrderedRange(options.headingMin, options.headingMax, "--heading-min/max");
  requireOrderedRange(options.heightMin, options.heightMax, "--height-min/max");
  requirePositiveSegmentRange(options.cmdVelSegmentMin, options.cmdVelSegmentMax, "--cmd-vel-segment-min/max");
  requirePositiveSegmentRange(options.waistSegmentMin, options.waistSegmentMax, "--waist-segment-min/max");
  requirePositiveSegmentRange(options.armSegmentMin, options.armSegmentMax, "--arm-segment-min/max");
  requirePositiveSegmentRange(options.wristSegmentMin, options.wristSegmentMax, "--wrist-segment-min/max");
  if (!std::isfinite(options.duration) || !std::isfinite(options.fps) ||
      !std::isfinite(options.duration * options.fps) || options.duration * options.fps > 10000000) {
    throw std::runtime_error("Duration/fps must be finite and limited to 10 million frames.");
  }
  if (options.noStepTransitions && options.stepConditions.empty())
    throw std::runtime_error("--no-step-transitions requires --step-condition.");
  if (!options.conditions.empty() && !options.stepConditions.empty())
    throw std::runtime_error("Use either --condition or --step-condition, not both.");
  if (!std::isfinite(options.doubleSupportFraction) || options.doubleSupportFraction <= 0 || options.doubleSupportFraction >= 1 ||
      !std::isfinite(options.footPositionWeight) || options.footPositionWeight <= 0)
    throw std::runtime_error("Double support fraction must be in (0,1) and foot position weight positive and finite.");
  if (!options.stepConditions.empty()) {
    StepConditioning schedule(options.stepConditions, static_cast<size_t>(std::floor(options.duration * options.fps)) + 1,
                              options.fps, options.plateauRatio, !options.noStepTransitions);
  }
  if (options.checkConditioning && options.conditions.empty() && options.stepConditions.empty()) {
    throw std::runtime_error("--check-conditioning requires --condition or --step-condition.");
  }
  if (!options.conditions.empty()) {
    MotionConditioning schedule(options.conditions, static_cast<size_t>(std::floor(options.duration * options.fps)) + 1,
                                options.fps, options.plateauRatio);
  }
  return options;
}

std::filesystem::path outputPathForMotion(const std::filesystem::path& requestedOutput, size_t motionIndex, size_t numMotions) {
  if (numMotions == 1) {
    return requestedOutput;
  }

  std::ostringstream suffix;
  suffix << "_" << std::setw(4) << std::setfill('0') << motionIndex;

  if (requestedOutput.has_extension()) {
    return requestedOutput.parent_path() / (requestedOutput.stem().string() + suffix.str() + requestedOutput.extension().string());
  }

  return requestedOutput / ("g1_random_mpc_reference" + suffix.str() + ".npz");
}

std::filesystem::path temporaryOutputPathForAttempt(const std::filesystem::path& outputPath, size_t attemptIndex) {
  std::ostringstream suffix;
  suffix << ".attempt_" << std::setw(3) << std::setfill('0') << attemptIndex << ".tmp";
  return outputPath.parent_path() / (outputPath.filename().string() + suffix.str());
}

void applyBaseCommandDeadband(vector4_t& command, scalar_t deadband) {
  if (deadband <= 0.0) {
    return;
  }
  for (const int index : {0, 1, 3}) {
    if (std::abs(command[index]) < deadband) {
      command[index] = 0.0;
    }
  }
}

SmoothScalarTrajectory makeRandomSmoothTrajectory(scalar_t duration,
                                                  scalar_t segmentMin,
                                                  scalar_t segmentMax,
                                                  scalar_t valueMin,
                                                  scalar_t valueMax,
                                                  scalar_t initialValue,
                                                  scalar_t holdPreviousValueProbability,
                                                  std::mt19937& rng) {
  std::uniform_real_distribution<scalar_t> segmentDist(segmentMin, segmentMax);
  std::uniform_real_distribution<scalar_t> valueDist(valueMin, valueMax);
  std::bernoulli_distribution holdDist(std::clamp(holdPreviousValueProbability, scalar_t(0.0), scalar_t(1.0)));

  SmoothScalarTrajectory trajectory;
  trajectory.times.push_back(0.0);
  trajectory.values.push_back(std::clamp(initialValue, valueMin, valueMax));

  scalar_t time = 0.0;
  while (time < duration) {
    time = std::min(duration, time + segmentDist(rng));
    trajectory.times.push_back(time);
    trajectory.values.push_back(holdDist(rng) ? trajectory.values.back() : valueDist(rng));
  }
  return trajectory;
}

StanceSchedule makeRandomStanceSchedule(
    scalar_t duration, scalar_t segmentMin, scalar_t segmentMax, scalar_t stanceProbability, std::mt19937& rng) {
  std::uniform_real_distribution<scalar_t> segmentDist(segmentMin, segmentMax);
  std::bernoulli_distribution stanceDist(std::clamp(stanceProbability, scalar_t(0.0), scalar_t(1.0)));

  StanceSchedule schedule;
  schedule.times.push_back(0.0);
  schedule.values.push_back(stanceDist(rng));

  scalar_t time = 0.0;
  while (time < duration) {
    time = std::min(duration, time + segmentDist(rng));
    schedule.times.push_back(time);
    schedule.values.push_back(stanceDist(rng));
  }
  return schedule;
}

HeadingSchedule makeRandomHeadingSchedule(scalar_t duration,
                                          scalar_t segmentMin,
                                          scalar_t segmentMax,
                                          scalar_t headingProbability,
                                          scalar_t headingMin,
                                          scalar_t headingMax,
                                          std::mt19937& rng) {
  std::uniform_real_distribution<scalar_t> segmentDist(segmentMin, segmentMax);
  std::uniform_real_distribution<scalar_t> targetDist(headingMin, headingMax);
  std::bernoulli_distribution headingDist(std::clamp(headingProbability, scalar_t(0.0), scalar_t(1.0)));

  HeadingSchedule schedule;
  schedule.times.push_back(0.0);
  schedule.active.push_back(headingDist(rng));
  schedule.targets.push_back(targetDist(rng));

  scalar_t time = 0.0;
  while (time < duration) {
    time = std::min(duration, time + segmentDist(rng));
    schedule.times.push_back(time);
    schedule.active.push_back(headingDist(rng));
    schedule.targets.push_back(targetDist(rng));
  }
  return schedule;
}

scalar_t wrapToPi(scalar_t angle) {
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

void throwIfNotFinite(const vector_t& vector, const std::string& name, size_t motionIndex, size_t frame, scalar_t time) {
  if (!vector.allFinite()) {
    std::ostringstream ss;
    ss << "Non-finite " << name << " at motion " << motionIndex << ", frame " << frame << ", t=" << time;
    throw std::runtime_error(ss.str());
  }
}

void throwIfUnexpectedSize(
    const vector_t& vector, Eigen::Index expectedSize, const std::string& name, size_t motionIndex, size_t frame, scalar_t time) {
  if (vector.size() != expectedSize) {
    std::ostringstream ss;
    ss << "Unexpected " << name << " size at motion " << motionIndex << ", frame " << frame << ", t=" << time << ": got " << vector.size()
       << ", expected " << expectedSize;
    throw std::runtime_error(ss.str());
  }
}

bool isWaistJoint(const std::string& jointName) {
  return jointName.find("waist") != std::string::npos;
}

bool isArmJoint(const std::string& jointName) {
  return jointName.find("shoulder") != std::string::npos || jointName.find("elbow") != std::string::npos ||
         jointName.find("wrist") != std::string::npos;
}

bool isWristJoint(const std::string& jointName) {
  return jointName.find("wrist") != std::string::npos;
}

std::vector<size_t> getRandomizedMpcJointIndices(const ModelSettings& modelSettings) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < modelSettings.mpcModelJointNames.size(); ++i) {
    const auto& name = modelSettings.mpcModelJointNames[i];
    if (isWaistJoint(name) || isArmJoint(name)) {
      indices.push_back(i);
    }
  }
  return indices;
}

std::pair<scalar_t, scalar_t> getJointLimits(const pinocchio::Model& model, const std::string& jointName) {
  const pinocchio::JointIndex jointId = model.getJointId(jointName);
  if (jointId == static_cast<pinocchio::JointIndex>(model.njoints)) {
    throw std::runtime_error("Joint not found in Pinocchio model: " + jointName);
  }
  const int qIndex = model.idx_qs[jointId];
  scalar_t lower = model.lowerPositionLimit[qIndex];
  scalar_t upper = model.upperPositionLimit[qIndex];
  if (!std::isfinite(lower) || !std::isfinite(upper) || lower >= upper) {
    lower = -1.0;
    upper = 1.0;
  }
  return {lower, upper};
}

std::vector<std::string> toStringVector(const std::vector<std::string>& names) {
  return names;
}

std::vector<std::string> clampBodyNames() {
  std::vector<std::string> names;
  names.reserve(kClampBodyNames.size());
  for (const char* name : kClampBodyNames) {
    names.emplace_back(name);
  }
  return names;
}

void appendFullJointState(MotionBuffers& buffers, const vector_t& fullJointPos, const vector_t& fullJointVel) {
  for (Eigen::Index i = 0; i < fullJointPos.size(); ++i) {
    buffers.jointPos.push_back(static_cast<float>(fullJointPos[i]));
  }
  for (Eigen::Index i = 0; i < fullJointVel.size(); ++i) {
    buffers.jointVel.push_back(static_cast<float>(fullJointVel[i]));
  }
}

void appendBodyPoses(MotionBuffers& buffers,
                     pinocchio::Model& model,
                     pinocchio::Data& data,
                     const std::vector<pinocchio::FrameIndex>& frameIds,
                     const vector6_t& basePose,
                     const vector_t& fullJointPos) {
  vector_t q(model.nq);
  q.setZero();
  q.head<6>() = basePose;
  q.tail(fullJointPos.size()) = fullJointPos;

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  for (const auto frameId : frameIds) {
    const auto& placement = data.oMf[frameId];
    const Eigen::Quaternion<scalar_t> quat(placement.rotation());
    buffers.bodyPos.push_back(static_cast<float>(placement.translation().x()));
    buffers.bodyPos.push_back(static_cast<float>(placement.translation().y()));
    buffers.bodyPos.push_back(static_cast<float>(placement.translation().z()));
    buffers.bodyQuat.push_back(static_cast<float>(quat.w()));
    buffers.bodyQuat.push_back(static_cast<float>(quat.x()));
    buffers.bodyQuat.push_back(static_cast<float>(quat.y()));
    buffers.bodyQuat.push_back(static_cast<float>(quat.z()));
  }
}

void appendContactData(MotionBuffers& buffers,
                       const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                       const vector_t& input,
                       const contact_flag_t& contactFlags) {
  if (contactFlags.size() != buffers.numContacts) {
    throw std::runtime_error("Contact flag dimension does not match configured contacts.");
  }

  for (size_t contactIndex = 0; contactIndex < buffers.numContacts; ++contactIndex) {
    const vector6_t wrench = mpcRobotModel.getContactWrench(input, contactIndex);
    for (Eigen::Index i = 0; i < wrench.size(); ++i) {
      buffers.contactWrench.push_back(static_cast<float>(wrench[i]));
    }
  }

  for (size_t contactIndex = 0; contactIndex < buffers.numContacts; ++contactIndex) {
    buffers.contactFlags.push_back(contactFlags[contactIndex] ? uint8_t{1} : uint8_t{0});
  }
}

void fillFiniteDifferenceBodyVelocities(MotionBuffers& buffers, scalar_t dt) {
  const size_t nFrames = buffers.numFrames;
  const size_t nBodies = buffers.numBodies;
  buffers.bodyLinVel.assign(nFrames * nBodies * 3, 0.0f);
  buffers.bodyAngVel.assign(nFrames * nBodies * 3, 0.0f);

  if (nFrames < 2) {
    return;
  }

  for (size_t t = 0; t + 1 < nFrames; ++t) {
    for (size_t b = 0; b < nBodies; ++b) {
      const size_t p0 = (t * nBodies + b) * 3;
      const size_t p1 = ((t + 1) * nBodies + b) * 3;
      for (size_t k = 0; k < 3; ++k) {
        buffers.bodyLinVel[p0 + k] = (buffers.bodyPos[p1 + k] - buffers.bodyPos[p0 + k]) / static_cast<float>(dt);
      }

      const size_t q0 = (t * nBodies + b) * 4;
      const size_t q1 = ((t + 1) * nBodies + b) * 4;
      Eigen::Quaternion<scalar_t> quat0(buffers.bodyQuat[q0], buffers.bodyQuat[q0 + 1], buffers.bodyQuat[q0 + 2], buffers.bodyQuat[q0 + 3]);
      Eigen::Quaternion<scalar_t> quat1(buffers.bodyQuat[q1], buffers.bodyQuat[q1 + 1], buffers.bodyQuat[q1 + 2], buffers.bodyQuat[q1 + 3]);
      quat0.normalize();
      quat1.normalize();
      Eigen::Quaternion<scalar_t> delta = quat1 * quat0.conjugate();
      if (delta.w() < 0.0) {
        delta.coeffs() *= -1.0;
      }
      Eigen::AngleAxis<scalar_t> angleAxis(delta);
      const vector3_t omega = angleAxis.axis() * angleAxis.angle() / dt;
      const size_t w0 = (t * nBodies + b) * 3;
      buffers.bodyAngVel[w0] = static_cast<float>(omega.x());
      buffers.bodyAngVel[w0 + 1] = static_cast<float>(omega.y());
      buffers.bodyAngVel[w0 + 2] = static_cast<float>(omega.z());
    }
  }

  const size_t lastOffset = (nFrames - 1) * nBodies * 3;
  const size_t prevOffset = (nFrames - 2) * nBodies * 3;
  std::copy_n(buffers.bodyLinVel.begin() + prevOffset, nBodies * 3, buffers.bodyLinVel.begin() + lastOffset);
  std::copy_n(buffers.bodyAngVel.begin() + prevOffset, nBodies * 3, buffers.bodyAngVel.begin() + lastOffset);
}

ModeSchedule makeConditionedModeSchedule(const MotionConditioning& conditions, const ModeSequenceTemplate& gait,
                                         scalar_t finalTime) {
  if (gait.switchingTimes.size() != 5 || gait.switchingTimes.front() != 0.0 || gait.switchingTimes.back() <= 0.0 ||
      gait.modeSequence.size() != 4 || gait.modeSequence[0] != LF || gait.modeSequence[1] != STANCE ||
      gait.modeSequence[2] != RF || gait.modeSequence[3] != STANCE) {
    throw std::runtime_error("Conditioned gait must have LF,STANCE,RF,STANCE phases and positive timing starting at zero.");
  }
  std::vector<double> phases;
  for (size_t i = 0; i < gait.switchingTimes.size(); ++i) {
    if (!std::isfinite(gait.switchingTimes[i]) || (i && gait.switchingTimes[i] <= gait.switchingTimes[i - 1])) {
      throw std::runtime_error("Conditioned gait switching times must be strictly increasing.");
    }
    phases.push_back(gait.switchingTimes[i] / gait.switchingTimes.back());
  }
  double shortestPhase = 1.0;
  for (size_t i = 1; i < phases.size(); ++i) shortestPhase = std::min(shortestPhase, phases[i] - phases[i - 1]);
  // ModeSchedule defaults to {0} (flight); replace it with initial double support.
  ModeSchedule result({}, {STANCE});
  size_t phaseIndex = 3;  // Begin in double support; never reset phase during command transitions.
  double phase = phases[phaseIndex];
  constexpr double integrationStep = 0.001;
  for (double t = 0.0; t < finalTime; ) {
    const double step = std::min(integrationStep, finalTime - t);
    const auto c = conditions.sample(t + 0.5 * step).command;
    const double increment = step * c.velocity / c.strideLength;
    if (!std::isfinite(increment) || increment > shortestPhase) {
      throw std::runtime_error("Requested velocity/stride cadence exceeds the 1 ms contact schedule resolution.");
    }
    double consumed = 0.0;
    while (phase + increment - consumed >= phases[phaseIndex + 1]) {
      const double toBoundary = phases[phaseIndex + 1] - phase;
      consumed += toBoundary;
      const double eventTime = t + step * consumed / increment;
      phaseIndex = (phaseIndex + 1) % gait.modeSequence.size();
      phase = phases[phaseIndex];
      result.eventTimes.push_back(eventTime);
      result.modeSequence.push_back(gait.modeSequence[phaseIndex]);
    }
    phase += increment - consumed;
    t += step;
  }
  // GaitSchedule uses a final stance sentinel when extending its scheduling window.
  result.eventTimes.push_back(finalTime);
  result.modeSequence.push_back(STANCE);
  return result;
}

void generateMotion(const Options& options, const std::filesystem::path& outputPath, uint32_t seed, size_t motionIndex) {
  std::cout << "[random_mpc_generator] motion " << motionIndex << " seed: " << seed << "\n";
  std::cout << "[random_mpc_generator] motion " << motionIndex << " output: " << outputPath.string() << "\n";

  std::mt19937 rng(seed);

  CentroidalMpcInterface interface(options.taskFile, options.urdfFile, options.referenceFile, true);
  interface.getSwitchedModelReferenceManagerPtr()->setArmSwingReferenceActive(false);

  std::optional<LipReference> lipReference;
  std::optional<FrameHeightCost> torsoHeightProbe;
  scalar_t torsoHeightTarget = 0.0;
  if (!options.lipReference.empty()) {
    lipReference.emplace(loadLipReference(options.lipReference));
    torsoHeightTarget = options.torsoOriginHeight
                            ? *options.torsoOriginHeight
                            : parseScalar(lipReference->metadata.at("requested_torso_origin_height"), "requested_torso_origin_height");
    torsoHeightProbe.emplace(interface.getPinocchioInterface(), interface.getMpcRobotModel(), "torso_link", torsoHeightTarget, 1.0);
    if (options.torsoHeightWeight > 0.0) {
      interface.getOptimalControlProblemRef().stateCostPtr->add(
          "torsoOriginHeightCost", std::make_unique<FrameHeightCost>(interface.getPinocchioInterface(), interface.getMpcRobotModel(),
                                                                     "torso_link", torsoHeightTarget, options.torsoHeightWeight));
    }
    std::cout << "[random_mpc_generator] torso_link origin height target " << torsoHeightTarget << " m, weight "
              << options.torsoHeightWeight << (options.torsoHeightWeight > 0.0 ? "" : " (disabled)") << "\n";
  }

  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(), interface.getInitializer());
  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  MPC_MRT_Interface mrt(mpc);
  mrt.initRollout(&interface.getRollout());

  const auto& mpcRobotModel = interface.getMpcRobotModel();
  const auto& modelSettings = interface.modelSettings();

  scalar_t defaultBaseHeight = 0.0;
  vector_t defaultMpcJointState(mpcRobotModel.getJointDim());
  loadData::loadCppDataType(options.referenceFile, "defaultBaseHeight", defaultBaseHeight);
  loadData::loadEigenMatrix(options.referenceFile, "defaultJointState", defaultMpcJointState);

  PinocchioInterface fullPinocchioInterface = createDefaultPinocchioInterface(options.urdfFile);
  pinocchio::Model fullModel = fullPinocchioInterface.getModel();
  pinocchio::Data fullData(fullModel);

  const auto randomizedMpcJointIndices = getRandomizedMpcJointIndices(modelSettings);
  std::unordered_map<size_t, SmoothScalarTrajectory> jointTrajectories;
  const bool lipMode = !options.lipReference.empty();
  if (lipMode) {
    // Nominal leg-joint targets: the calibrated posture the LIP CoM height came from.
    size_t legJoints = 0;
    for (size_t i = 0; i < modelSettings.mpcModelJointNames.size(); ++i) {
      const auto& name = modelSettings.mpcModelJointNames[i];
      if (name.find("hip") == std::string::npos && name.find("knee") == std::string::npos && name.find("ankle") == std::string::npos) {
        continue;
      }
      const auto it = lipReference->initialJointPos.find(name);
      if (it == lipReference->initialJointPos.end()) throw std::runtime_error("LIP reference posture lacks leg joint " + name);
      defaultMpcJointState[i] = it->second;
      ++legJoints;
    }
    if (legJoints != 12) throw std::runtime_error("Expected 12 MPC leg joints, found " + std::to_string(legJoints));
  }
  const bool neutralUpperBody = options.upperBodyMode == "neutral" ||
      (options.upperBodyMode == "auto" && (!options.stepConditions.empty() || lipMode));
  std::cout << "[random_mpc_generator] upper-body targets: " << (neutralUpperBody ? "neutral" : "random") << "\n";
  for (const size_t mpcJointIndex : randomizedMpcJointIndices) {
    // Leave the configured nominal waist/arm/wrist posture as the MPC target.
    // Do not clamp exported joints: balance dynamics must still be solved.
    if (neutralUpperBody) continue;
    const std::string& jointName = modelSettings.mpcModelJointNames[mpcJointIndex];
    const auto [lowerRaw, upperRaw] = getJointLimits(fullModel, jointName);
    scalar_t lower = lowerRaw + options.armLimitMargin * (upperRaw - lowerRaw);
    scalar_t upper = upperRaw - options.armLimitMargin * (upperRaw - lowerRaw);
    scalar_t segmentMin = options.armSegmentMin;
    scalar_t segmentMax = options.armSegmentMax;
    if (isWaistJoint(jointName)) {
      lower = std::max(lower, -M_PI / 2.0);
      upper = std::min(upper, M_PI / 2.0);
      segmentMin = options.waistSegmentMin;
      segmentMax = options.waistSegmentMax;
    } else if (isWristJoint(jointName)) {
      segmentMin = options.wristSegmentMin;
      segmentMax = options.wristSegmentMax;
    }
    jointTrajectories.emplace(mpcJointIndex,
                              makeRandomSmoothTrajectory(options.duration, segmentMin, segmentMax, lower, upper,
                                                         defaultMpcJointState[mpcJointIndex], options.upperBodyFixedProbability, rng));
    std::cout << "[random_mpc_generator] sampling " << jointName << " in [" << lower << ", " << upper << "] every [" << segmentMin << ", "
              << segmentMax << "] s\n";
  }

  const auto vxTrajectory = makeRandomSmoothTrajectory(options.duration, options.cmdVelSegmentMin, options.cmdVelSegmentMax, options.vxMin,
                                                       options.vxMax, 0.0, 0.0, rng);
  const auto vyTrajectory = makeRandomSmoothTrajectory(options.duration, options.cmdVelSegmentMin, options.cmdVelSegmentMax, options.vyMin,
                                                       options.vyMax, 0.0, 0.0, rng);
  const auto yawRateTrajectory = makeRandomSmoothTrajectory(options.duration, options.cmdVelSegmentMin, options.cmdVelSegmentMax,
                                                            options.yawRateMin, options.yawRateMax, 0.0, 0.0, rng);
  const auto heightTrajectory = makeRandomSmoothTrajectory(options.duration, options.cmdVelSegmentMin, options.cmdVelSegmentMax,
                                                           options.heightMin, options.heightMax, defaultBaseHeight, 0.0, rng);
  const auto stanceSchedule =
      makeRandomStanceSchedule(options.duration, options.cmdVelSegmentMin, options.cmdVelSegmentMax, options.stanceProbability, rng);
  const auto headingSchedule = makeRandomHeadingSchedule(options.duration, options.cmdVelSegmentMin, options.cmdVelSegmentMax,
                                                         options.headingProbability, options.headingMin, options.headingMax, rng);

  const auto gaitMap = getGaitMap(options.gaitFile);
  const bool commandConditioned = !options.conditions.empty() || !options.stepConditions.empty();
  // LIP mode follows the conditioned path: no gait switching, random heading, or velocity filtering.
  const bool conditioned = commandConditioned || lipMode;
  const std::vector<ProceduralMpcMotionManager::GaitModeStateConfig> gaitModeStates{
      {"stance", -0.1, 0.1, -0.1, 0.1, 10.0, 10.0}, {"slow_walk", 0.05, 0.3, 0.05, 0.2, 0.05, 0.05},
      {"walk", 0.25, 0.5, 0.15, 0.35, 0.05, 0.05},  {"slower_trot", 0.45, 0.7, 0.3, 0.55, 0.1, 0.1},
      {"slow_trot", 0.65, 0.9, 0.5, 0.7, 0.2, 0.2}, {"trot", 0.8, 1.3, 0.65, 10.0, 0.2, 0.2},
      {"run", 1.2, 10.0, 0.65, 10.0, 0.2, 0.2},
  };
  for (const auto& gaitModeState : gaitModeStates) {
    if (!conditioned && gaitMap.find(gaitModeState.gaitCommand) == gaitMap.end()) {
      throw std::runtime_error("Requested gait `" + gaitModeState.gaitCommand + "` is not present in " + options.gaitFile);
    }
  }
  auto gaitSchedulePtr = interface.getSwitchedModelReferenceManagerPtr()->getGaitSchedule();
  BreakFrequencyAlphaFilter velocityCommandFilter(5, vector4_t::Zero());
  size_t currentGaitMode = 0;
  std::string currentGaitCommand = gaitModeStates[currentGaitMode].gaitCommand;
  std::string lastGaitCommand = currentGaitCommand;
  scalar_t lastGaitChangeTime = 0.0;
  const size_t waistYawJointIndex = mpcRobotModel.getJointIndex("waist_yaw_joint");
  const size_t waistRollJointIndex = mpcRobotModel.getJointIndex("waist_roll_joint");
  const size_t waistPitchJointIndex = mpcRobotModel.getJointIndex("waist_pitch_joint");

  CentroidalMpcTargetTrajectoriesCalculator targetCalculator(options.referenceFile, mpcRobotModel, interface.getPinocchioInterface(),
                                                             interface.getCentroidalModelInfo(), interface.mpcSettings().timeHorizon_);

  targetCalculator.setVelocityReferenceSmoothingEnabled(!conditioned);

  SystemObservation observation;
  observation.time = 0.0;
  observation.state = interface.getInitialState();
  if (lipReference) {
    // Start in the calibrated posture: base pose plus every MPC joint, zero momentum.
    vector_t joints = mpcRobotModel.getJointAngles(observation.state);
    for (size_t i = 0; i < modelSettings.mpcModelJointNames.size(); ++i) {
      const auto it = lipReference->initialJointPos.find(modelSettings.mpcModelJointNames[i]);
      if (it == lipReference->initialJointPos.end()) {
        throw std::runtime_error("LIP reference posture lacks MPC joint " + modelSettings.mpcModelJointNames[i]);
      }
      joints[i] = it->second;
    }
    observation.state.head<6>().setZero();
    mpcRobotModel.setBasePose(observation.state, lipReference->initialBasePose);
    mpcRobotModel.setJointAngles(observation.state, joints);
    std::cout << "[random_mpc_generator] initial state: calibrated posture, pelvis z " << lipReference->initialBasePose[2]
              << " m, torso_link origin z " << torsoHeightProbe->height(observation.state) << " m\n";
  }
  observation.input = vector_t::Zero(interface.getCentroidalModelInfo().inputDim);
  observation.mode = ModeNumber::STANCE;

  WalkingVelocityCommand initialVelocityTarget;
  initialVelocityTarget.desired_pelvis_height = defaultBaseHeight;
  for (int i = 0; i < 20; ++i) {
    targetCalculator.commandedVelocityToTargetTrajectories(initialVelocityTarget, observation.time, observation.state);
  }

  const std::vector<std::string> bodyNames = clampBodyNames();
  std::vector<pinocchio::FrameIndex> bodyFrameIds;
  bodyFrameIds.reserve(bodyNames.size());
  for (const auto& bodyName : bodyNames) {
    const auto frameId = fullModel.getFrameId(bodyName);
    if (frameId == static_cast<pinocchio::FrameIndex>(fullModel.nframes)) {
      throw std::runtime_error("Body frame not found in full Pinocchio model: " + bodyName);
    }
    bodyFrameIds.push_back(frameId);
  }

  vector_t defaultFullJointState = vector_t::Zero(modelSettings.full_joint_dim);
  defaultFullJointState = mpcRobotModel.getFullModelJointAngles(defaultMpcJointState, defaultFullJointState);

  MotionBuffers buffers;
  buffers.numJoints = modelSettings.full_joint_dim;
  buffers.numBodies = bodyNames.size();
  buffers.numContacts = modelSettings.contactNames6DoF.size();
  buffers.fps = {options.fps};

  const scalar_t dt = 1.0 / options.fps;
  const size_t numFrames = static_cast<size_t>(std::floor(options.duration * options.fps)) + 1;

  // Construct only in conditioned mode; random mode keeps its existing behavior.
  std::optional<MotionConditioning> conditionSchedule;
  if (!options.conditions.empty()) {
    conditionSchedule.emplace(options.conditions, numFrames, options.fps, options.plateauRatio);
    gaitSchedulePtr->updateModeSchedule(makeConditionedModeSchedule(*conditionSchedule, gaitMap.at(options.conditionedGait),
        options.duration + 4.0 * interface.mpcSettings().timeHorizon_ + 1.0));
  }
  std::optional<StepConditioning> stepSchedule;
  std::shared_ptr<const StepReferencePlan> stepPlan;
  if (!options.stepConditions.empty()) {
    stepSchedule.emplace(options.stepConditions, numFrames, options.fps, options.plateauRatio, !options.noStepTransitions);
    // Anchor to actual initial contact frames, once, before advancing the rollout.
    auto initialPinocchio = interface.getPinocchioInterface();
    const auto q = mpcRobotModel.getGeneralizedCoordinates(observation.state);
    pinocchio::forwardKinematics(initialPinocchio.getModel(), initialPinocchio.getData(), q);
    pinocchio::updateFramePlacements(initialPinocchio.getModel(), initialPinocchio.getData());
    std::array<vector3_t, 2> feet;
    for (size_t foot = 0; foot < 2; ++foot) {
      const auto id = initialPinocchio.getModel().getFrameId(modelSettings.contactNames[foot]);
      feet[foot] = initialPinocchio.getData().oMf[id].translation();
    }
    stepPlan = std::make_shared<StepReferencePlan>(feet, mpcRobotModel.getBasePose(observation.state)[3],
        options.duration + 4.0 * interface.mpcSettings().timeHorizon_ + 1.0, options.doubleSupportFraction,
        [&](double t) { return stepSchedule->sample(t).command; }, options.footPositionWeight);
    interface.getSwitchedModelReferenceManagerPtr()->setStepReferencePlan(stepPlan);
    gaitSchedulePtr->updateModeSchedule(stepPlan->modeSchedule());
    for (const auto& step : stepPlan->steps()) {
      buffers.plannedSteps.insert(buffers.plannedSteps.end(), {step.start, step.liftOff, step.touchDown,
          static_cast<double>(step.foot), step.command.length, step.command.duration, step.to.x(), step.to.y()});
    }
  }
  PinocchioInterface comPinocchio = interface.getPinocchioInterface();
  const auto measuredCom = [&](const vector_t& state) -> vector3_t {
    return pinocchio::centerOfMass(comPinocchio.getModel(), comPinocchio.getData(), mpcRobotModel.getGeneralizedCoordinates(state));
  };
  std::shared_ptr<const LipReferencePlan> lipPlan;
  if (lipMode) {
    const LipReference& lip = *lipReference;
    std::vector<LipSample> samples;
    samples.reserve(lip.size());
    for (size_t k = 0; k < lip.size(); ++k) {
      samples.push_back({lip.time[k], lip.contact[k], lip.comPos[k], lip.comVel[k], lip.footPos[k], lip.footVel[k]});
    }
    lipPlan = std::make_shared<LipReferencePlan>(std::move(samples), lip.heading, options.footPositionWeight);
    if (options.duration + interface.mpcSettings().timeHorizon_ > lipPlan->finalTime() + 1e-9) {
      throw std::runtime_error("LIP reference ends at " + std::to_string(lipPlan->finalTime()) + " s; duration + MPC horizon is " +
                               std::to_string(options.duration + interface.mpcSettings().timeHorizon_) + " s.");
    }
    // Contact timing (support mask), foot targets, and foot height all come from the LIP samples.
    interface.getSwitchedModelReferenceManagerPtr()->setStepReferencePlan(lipPlan);
    gaitSchedulePtr->updateModeSchedule(lipPlan->modeSchedule());

    auto initialPinocchio = interface.getPinocchioInterface();
    pinocchio::forwardKinematics(initialPinocchio.getModel(), initialPinocchio.getData(),
                                 mpcRobotModel.getGeneralizedCoordinates(observation.state));
    pinocchio::updateFramePlacements(initialPinocchio.getModel(), initialPinocchio.getData());
    std::cout << std::setprecision(6) << std::fixed << "[random_mpc_generator] LIP plan: " << lipPlan->samples().size() << " samples, "
              << lipPlan->modeSchedule().eventTimes.size() << " contact events, heading " << lipPlan->heading() << "\n"
              << "[random_mpc_generator] initial MPC CoM - LIP CoM: "
              << (measuredCom(observation.state) - lipPlan->com(0.0).first).transpose() << "\n";
    for (size_t foot = 0; foot < 2; ++foot) {
      const auto id = initialPinocchio.getModel().getFrameId(modelSettings.contactNames[foot]);
      std::cout << "[random_mpc_generator] initial MPC " << modelSettings.contactNames[foot] << " - LIP foot: "
                << (initialPinocchio.getData().oMf[id].translation() - lipPlan->sample(foot, 0.0).position).transpose() << "\n";
    }
  }
  // Initialize with the same contact schedule and foot anchors used by rollout.
  TargetTrajectories initTargetTrajectories({0.0}, {observation.state}, {observation.input});
  mrt.resetMpcNode(initTargetTrajectories);
  mrt.setCurrentObservation(observation);

  size_t attempts = 0;
  while (!mrt.initialPolicyReceived()) {
    mrt.advanceMpc();
    mrt.updatePolicy();
    if (++attempts > 20) {
      throw std::runtime_error("Failed to receive the initial MPC policy.");
    }
  }

  for (size_t frame = 0; frame < numFrames; ++frame) {
    const scalar_t time = observation.time;

    vector_t sampledMpcJointState = defaultMpcJointState;
    for (const auto& [jointIndex, trajectory] : jointTrajectories) {
      sampledMpcJointState[jointIndex] = trajectory.value(time);
    }

    const bool forceStance = !conditioned && stanceSchedule.value(time);
    auto condition = conditionSchedule ? conditionSchedule->sample(time) : ConditionSample{};
    if (stepPlan) {
      const auto applied = stepPlan->stepAt(time);
      const auto requested = stepSchedule->sample(applied.start);
      // Legacy fields retain their documented velocity/full-stride meaning.
      condition = {{applied.command.length / applied.command.duration, 2.0 * applied.command.length},
                   {requested.target.length / requested.target.duration, 2.0 * requested.target.length},
                   requested.stage, requested.transition};
    }
    vector4_t velocityTarget;
    velocityTarget << vxTrajectory.value(time), vyTrajectory.value(time), heightTrajectory.value(time), yawRateTrajectory.value(time);
    if (forceStance) {
      velocityTarget[0] = 0.0;
      velocityTarget[1] = 0.0;
      velocityTarget[3] = 0.0;
    }
    WalkingVelocityCommand filteredCommand;
    filteredCommand.setFromVector(velocityCommandFilter.getFilteredVector(velocityTarget));
    const auto [useHeading, headingTarget] = headingSchedule.value(time);
    if (!conditioned && !forceStance && useHeading) {
      // Base pose layout is [x, y, z, yaw, pitch, roll].
      const scalar_t baseYaw = mpcRobotModel.getBasePose(observation.state)[3];
      const scalar_t yawRate = options.headingControlStiffness * wrapToPi(headingTarget - baseYaw);
      filteredCommand.angular_velocity_z = std::clamp(yawRate, options.yawRateMin, options.yawRateMax);
    }
    if (forceStance) {
      filteredCommand.linear_velocity_x = 0.0;
      filteredCommand.linear_velocity_y = 0.0;
      filteredCommand.angular_velocity_z = 0.0;
    }
    vector4_t filteredVelCommand = filteredCommand.toVector();
    if (conditioned) {
      // The schedule already provides a C2 ramp. Avoid filtering/deadband distortion of labels and cadence.
      filteredVelCommand << condition.command.velocity, 0.0, defaultBaseHeight, 0.0;
    } else {
      applyBaseCommandDeadband(filteredVelCommand, options.baseCommandDeadband);
    }
    filteredCommand.setFromVector(filteredVelCommand);
    filteredCommand.desired_waist_yaw = sampledMpcJointState[waistYawJointIndex];
    filteredCommand.desired_waist_roll = sampledMpcJointState[waistRollJointIndex];
    filteredCommand.desired_waist_pitch = sampledMpcJointState[waistPitchJointIndex];
    const vector6_t baseVelocity = mpcRobotModel.getBaseComVelocity(observation.state);
    auto currentCfg = gaitModeStates[currentGaitMode];

    if (forceStance && currentGaitCommand != "stance") {
      currentGaitMode = 0;
      currentCfg = gaitModeStates[currentGaitMode];
      currentGaitCommand = currentCfg.gaitCommand;
      lastGaitChangeTime = time;
      std::cout << "[random_mpc_generator] forcing stance at t=" << std::fixed << std::setprecision(2) << time << "\n";
    } else if (!conditioned && !forceStance && time > lastGaitChangeTime + 0.2) {
      if (currentGaitMode + 1 < gaitModeStates.size() &&
          ProceduralMpcMotionManager::transitionToFasterGait(filteredVelCommand, baseVelocity, currentCfg)) {
        ++currentGaitMode;
        currentCfg = gaitModeStates[currentGaitMode];
        currentGaitCommand = currentCfg.gaitCommand;
        lastGaitChangeTime = time;
        std::cout << "[random_mpc_generator] increasing to gait `" << currentGaitCommand << "` at t=" << std::fixed << std::setprecision(2)
                  << time << "\n";
      } else if (currentGaitMode > 0 && ProceduralMpcMotionManager::transitionToSlowerGait(filteredVelCommand, baseVelocity, currentCfg)) {
        --currentGaitMode;
        currentCfg = gaitModeStates[currentGaitMode];
        currentGaitCommand = currentCfg.gaitCommand;
        lastGaitChangeTime = time;
        std::cout << "[random_mpc_generator] decreasing to gait `" << currentGaitCommand << "` at t=" << std::fixed << std::setprecision(2)
                  << time << "\n";
      }
    }

    if (!conditioned && currentGaitCommand != lastGaitCommand) {
      GaitScheduleUpdater::updateGaitSchedule(gaitSchedulePtr, gaitMap.at(currentGaitCommand), time,
                                              time + interface.mpcSettings().timeHorizon_);
      lastGaitCommand = currentGaitCommand;
    }

    targetCalculator.setTargetJointState(sampledMpcJointState);
    TargetTrajectories targetTrajectories =
        targetCalculator.commandedVelocityToTargetTrajectories(filteredCommand, time, observation.state);
    if (stepPlan) {
      for (auto& targetState : targetTrajectories.stateTrajectory) {
        targetState[mpcRobotModel.getBaseStartindex() + 3] = stepPlan->heading();
      }
    }
    if (lipPlan) {
      // CoM tracking: knots at the current time, every LIP sample inside the horizon, and the horizon end, so
      // the solver's linear interpolation follows the samples at any query time. Normalized linear momentum
      // is the CoM velocity. The base target is the LIP CoM minus the current CoM-to-base offset; joint
      // targets stay the nominal ones from the calculator.
      const scalar_t horizonEnd = time + interface.mpcSettings().timeHorizon_;
      const vector_t nominalState = targetTrajectories.stateTrajectory.front();
      const vector_t nominalInput = targetTrajectories.inputTrajectory.front();
      const vector3_t comToBase = mpcRobotModel.getBasePose(observation.state).head<3>() - measuredCom(observation.state);
      const size_t base = mpcRobotModel.getBaseStartindex();
      scalar_array_t knots{time};
      for (const auto& sample : lipPlan->samples()) {
        if (sample.time > time + 1e-9 && sample.time < horizonEnd - 1e-9) knots.push_back(sample.time);
      }
      knots.push_back(horizonEnd);
      TargetTrajectories lipTargets;
      for (const scalar_t knot : knots) {
        const auto [comPosition, comVelocity] = lipPlan->com(knot);
        vector_t state = nominalState;
        state.head<6>().setZero();
        state.head<3>() = comVelocity;
        state.segment<3>(base) = comPosition + comToBase;
        state.segment<3>(base + 3) << lipPlan->heading(), 0.0, 0.0;
        lipTargets.timeTrajectory.push_back(knot);
        lipTargets.stateTrajectory.push_back(std::move(state));
        lipTargets.inputTrajectory.push_back(nominalInput);
      }
      targetTrajectories = std::move(lipTargets);
    }
    mrt.getReferenceManager().setTargetTrajectories(targetTrajectories);

    mrt.setCurrentObservation(observation);
    mrt.advanceMpc();
    mrt.updatePolicy();

    vector_t nextState;
    vector_t nextInput;
    size_t nextMode = observation.mode;
    mrt.rolloutPolicy(time, observation.state, dt, nextState, nextInput, nextMode);
    throwIfUnexpectedSize(nextState, observation.state.size(), "rollout state", motionIndex, frame, time);
    throwIfUnexpectedSize(nextInput, observation.input.size(), "rollout input", motionIndex, frame, time);
    throwIfNotFinite(nextState, "rollout state", motionIndex, frame, time);
    throwIfNotFinite(nextInput, "rollout input", motionIndex, frame, time);

    observation.time = time + dt;
    observation.state = std::move(nextState);
    observation.input = std::move(nextInput);
    observation.mode = nextMode;

    const vector_t fullJointPos =
        mpcRobotModel.getFullModelJointAngles(mpcRobotModel.getJointAngles(observation.state), defaultFullJointState);
    vector_t fullJointVel = vector_t::Zero(modelSettings.full_joint_dim);
    fullJointVel =
        mpcRobotModel.getFullModelJointAngles(mpcRobotModel.getJointVelocities(observation.state, observation.input), fullJointVel);

    appendFullJointState(buffers, fullJointPos, fullJointVel);
    appendBodyPoses(buffers, fullModel, fullData, bodyFrameIds, mpcRobotModel.getBasePose(observation.state), fullJointPos);
    appendContactData(buffers, mpcRobotModel, observation.input,
                      interface.getSwitchedModelReferenceManagerPtr()->getContactFlags(observation.time));
    if (lipPlan) {
      const auto [comPosition, comVelocity] = lipPlan->com(observation.time);
      const auto contact = lipPlan->contactFlags(observation.time);
      const vector3_t com = measuredCom(observation.state);
      for (size_t axis = 0; axis < 3; ++axis) {
        buffers.lipComReference.push_back(static_cast<float>(comPosition[axis]));
        buffers.lipComVelReference.push_back(static_cast<float>(comVelocity[axis]));
        buffers.comPos.push_back(static_cast<float>(com[axis]));
      }
      for (size_t foot = 0; foot < 2; ++foot) {
        const auto ref = lipPlan->sample(foot, observation.time);
        for (size_t axis = 0; axis < 3; ++axis) {
          buffers.lipFootReference.push_back(static_cast<float>(ref.position[axis]));
          buffers.lipFootVelReference.push_back(static_cast<float>(ref.velocity[axis]));
        }
        buffers.lipContact.push_back(contact[foot] ? 1 : 0);
      }
    }
    if (commandConditioned) {
      buffers.conditioning.insert(buffers.conditioning.end(),
                                  {static_cast<float>(condition.command.velocity), static_cast<float>(condition.command.strideLength)});
      buffers.conditioningTarget.insert(buffers.conditioningTarget.end(),
                                  {static_cast<float>(condition.target.velocity), static_cast<float>(condition.target.strideLength)});
      for (int k = 0; k < 4; ++k) buffers.commandedVelocity.push_back(static_cast<float>(filteredVelCommand[k]));
      buffers.transition.push_back(condition.transition ? 1 : 0);
      buffers.conditionStage.push_back(static_cast<int32_t>(condition.stage));
      buffers.commandTime.push_back(time);
      buffers.sampleTime.push_back(observation.time);
    }
    if (stepPlan) {
      const auto applied = stepPlan->stepAt(time).command;
      const auto requested = stepSchedule->sample(time);
      buffers.stepConditioning.insert(buffers.stepConditioning.end(), {static_cast<float>(applied.length), static_cast<float>(applied.duration)});
      buffers.stepConditioningRequested.insert(buffers.stepConditioningRequested.end(), {static_cast<float>(requested.command.length), static_cast<float>(requested.command.duration)});
      buffers.stepConditioningTarget.insert(buffers.stepConditioningTarget.end(), {static_cast<float>(requested.target.length), static_cast<float>(requested.target.duration)});
      for (size_t foot = 0; foot < 2; ++foot) {
        const auto ref = stepPlan->sample(foot, observation.time);
        for (size_t axis = 0; axis < 2; ++axis) {
          buffers.footReferencePos.push_back(static_cast<float>(ref.position[axis]));
          buffers.footReferenceVel.push_back(static_cast<float>(ref.velocity[axis]));
        }
      }
    }
    ++buffers.numFrames;

    if (frame % 25 == 0) {
      std::cout << "[random_mpc_generator] frame " << frame << "/" << (numFrames - 1) << " time=" << std::fixed << std::setprecision(2)
                << observation.time << "\n";
    }
  }

  fillFiniteDifferenceBodyVelocities(buffers, dt);
  saveClampNpz(outputPath, buffers, toStringVector(modelSettings.fullJointNames), bodyNames, modelSettings.contactNames6DoF);

  std::cout << "[random_mpc_generator] saved " << buffers.numFrames << " frames to " << outputPath.string() << "\n";
}

int runGenerationAttempt(const Options& options, const std::filesystem::path& outputPath, uint32_t seed, size_t motionIndex) {
  std::cout.flush();
  std::cerr.flush();

  const pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error("Failed to fork a random MPC generation attempt.");
  }

  if (pid == 0) {
    try {
      generateMotion(options, outputPath, seed, motionIndex);
      std::cout.flush();
      std::cerr.flush();
      std::_Exit(EXIT_SUCCESS);
    } catch (const std::exception& e) {
      std::cerr << "[random_mpc_generator] child attempt failed with seed " << seed << ": " << e.what() << "\n";
      std::cout.flush();
      std::cerr.flush();
      std::_Exit(EXIT_FAILURE);
    } catch (...) {
      std::cerr << "[random_mpc_generator] child attempt failed with seed " << seed << ": unknown exception\n";
      std::cout.flush();
      std::cerr.flush();
      std::_Exit(EXIT_FAILURE);
    }
  }

  int status = 0;
  while (waitpid(pid, &status, 0) == -1) {
    if (errno == EINTR) {
      continue;
    }
    throw std::runtime_error("Failed while waiting for a random MPC generation attempt.");
  }
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  Options options = parseOptions(argc, argv);  // LIP mode may set the default duration below.
  if (!options.lipReference.empty()) {
    const LipReference lip = loadLipReference(options.lipReference);
    scalar_t horizon = 0.0;
    loadData::loadCppDataType(options.taskFile, "mpc.timeHorizon", horizon);
    if (!(lip.duration() > horizon)) {
      throw std::runtime_error("LIP reference is shorter than the MPC horizon (" + std::to_string(horizon) + " s).");
    }
    // Every MPC horizon must stay inside the samples.
    if (!options.durationGiven) options.duration = lip.duration() - horizon;
    if (options.duration + horizon > lip.duration() + 1e-9) {
      throw std::runtime_error("--duration " + std::to_string(options.duration) + " + MPC horizon " + std::to_string(horizon) +
                               " s exceeds the LIP reference duration " + std::to_string(lip.duration()) + " s.");
    }
    std::cout << "[random_mpc_generator] LIP reference: " << options.lipReference << " (" << lip.size() << " samples, dt " << lip.dt
              << " s, " << lip.duration() << " s, heading " << lip.heading << "); rollout duration " << options.duration << " s\n";
  }
  if (options.checkConditioning && !options.stepConditions.empty()) {
    const StepConditioning schedule(options.stepConditions,
        static_cast<size_t>(std::floor(options.duration * options.fps)) + 1, options.fps, options.plateauRatio, !options.noStepTransitions);
    const StepReferencePlan plan({vector3_t(0, 0.1, 0), vector3_t(0, -0.1, 0)}, 0, options.duration,
        options.doubleSupportFraction, [&](double t) { return schedule.sample(t).command; }, options.footPositionWeight);
    std::cout << "Step conditioning check only (no MPC, no output files). Synthetic feet for schedule inspection.\n";
    for (const auto& step : plan.steps()) {
      std::cout << "foot=" << step.foot << " liftoff=" << step.liftOff << " touchdown=" << step.touchDown
                << " length=" << step.command.length << " duration=" << step.command.duration
                << " nominal_velocity=" << step.command.length / step.command.duration
                << " target_xy=" << step.to.head<2>().transpose() << "\n";
    }
    return 0;
  }
  if (options.checkConditioning) {
    const size_t frames = static_cast<size_t>(std::floor(options.duration * options.fps)) + 1;
    const MotionConditioning schedule(options.conditions, frames, options.fps, options.plateauRatio);
    const auto gaitMap = getGaitMap(options.gaitFile);
    const auto modes = makeConditionedModeSchedule(schedule, gaitMap.at(options.conditionedGait), options.duration + 5.0);
    std::vector<size_t> holds(options.conditions.size(), 0);
    size_t transitions = 0;
    for (size_t i = 0; i < frames; ++i) {
      const auto c = schedule.sample(static_cast<double>(i) / options.fps);
      if (c.transition) ++transitions; else ++holds[c.stage];
    }
    std::cout << "Conditioning check only (no MPC, no output files): " << frames << " frames, " << transitions << " transition frames\n";
    for (size_t i = 0; i < holds.size(); ++i) {
      const auto c = options.conditions[i];
      std::cout << "stage " << i << ": velocity=" << c.velocity << " stride=" << c.strideLength
                << " cycle_s=" << c.strideLength / c.velocity << " steps_per_min=" << 120.0 * c.velocity / c.strideLength
                << " plateau_frames=" << holds[i] << "\n";
    }
    size_t cadenceChecks = 0;
    const auto& gait = gaitMap.at(options.conditionedGait);
    for (size_t i = 0; i + 1 < modes.eventTimes.size(); ++i) {
      if (!(modes.eventTimes[i + 1] > modes.eventTimes[i])) {
        throw std::runtime_error("Contact schedule events must be strictly increasing.");
      }
      const auto a = schedule.sample(modes.eventTimes[i] + 1e-6);
      const auto b = schedule.sample(modes.eventTimes[i + 1] - 1e-6);
      // The final sentinel can truncate a phase; only check complete phase intervals.
      if (i + 2 < modes.eventTimes.size() && !a.transition && !b.transition && a.stage == b.stage) {
        const size_t phase = i % gait.modeSequence.size();
        const double phaseFraction = (gait.switchingTimes[phase + 1] - gait.switchingTimes[phase]) / gait.switchingTimes.back();
        const double expected = phaseFraction * a.command.strideLength / a.command.velocity;
        if (std::abs(modes.eventTimes[i + 1] - modes.eventTimes[i] - expected) > 0.002) {
          throw std::runtime_error("Conditioned contact cadence does not match velocity/stride.");
        }
        ++cadenceChecks;
      }
    }
    std::cout << "Contact schedule validated: " << modes.eventTimes.size() << " events, " << cadenceChecks
              << " plateau phase durations checked\n";
    return 0;
  }

  std::cout << "[random_mpc_generator] task file: " << options.taskFile << "\n";
  std::cout << "[random_mpc_generator] reference file: " << options.referenceFile << "\n";
  std::cout << "[random_mpc_generator] urdf file: " << options.urdfFile << "\n";
  std::cout << "[random_mpc_generator] gait file: " << options.gaitFile << "\n";
  std::cout << "[random_mpc_generator] output: " << options.output << "\n";
  std::cout << "[random_mpc_generator] num motions: " << options.numMotions << "\n";
  std::cout << "[random_mpc_generator] max attempts per motion: " << options.maxAttemptsPerMotion << "\n";

  for (size_t motionIndex = 0; motionIndex < options.numMotions; ++motionIndex) {
    const auto outputPath = outputPathForMotion(options.output, motionIndex, options.numMotions);
    bool generated = false;
    for (size_t attemptIndex = 0; attemptIndex < options.maxAttemptsPerMotion; ++attemptIndex) {
      const uint32_t attemptSeed = options.seed + static_cast<uint32_t>(motionIndex * options.maxAttemptsPerMotion + attemptIndex);
      const auto temporaryOutputPath = temporaryOutputPathForAttempt(outputPath, attemptIndex);
      std::filesystem::remove(temporaryOutputPath);

      const int attemptStatus = runGenerationAttempt(options, temporaryOutputPath, attemptSeed, motionIndex);
      if (WIFEXITED(attemptStatus) && WEXITSTATUS(attemptStatus) == EXIT_SUCCESS) {
        std::filesystem::remove(outputPath);
        std::filesystem::rename(temporaryOutputPath, outputPath);
        std::cout << "[random_mpc_generator] motion " << motionIndex << " accepted after attempt " << (attemptIndex + 1) << "/"
                  << options.maxAttemptsPerMotion << ": " << outputPath.string() << "\n";
        generated = true;
        break;
      } else {
        std::filesystem::remove(temporaryOutputPath);
        std::cerr << "[random_mpc_generator] motion " << motionIndex << " attempt " << (attemptIndex + 1) << "/"
                  << options.maxAttemptsPerMotion << " failed with seed " << attemptSeed;
        if (WIFSIGNALED(attemptStatus)) {
          std::cerr << " due to signal " << WTERMSIG(attemptStatus);
        } else if (WIFEXITED(attemptStatus)) {
          std::cerr << " with exit code " << WEXITSTATUS(attemptStatus);
        }
        std::cerr << "\n";
      }
    }

    if (!generated) {
      std::ostringstream ss;
      ss << "Failed to generate motion " << motionIndex << " after " << options.maxAttemptsPerMotion << " attempts.";
      throw std::runtime_error(ss.str());
    }
  }
  return 0;
}
