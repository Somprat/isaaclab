#pragma once

#include <array>

#include <humanoid_common_mpc/common/Types.h>
#include <ocs2_core/reference/ModeSchedule.h>

namespace ocs2::humanoid {

struct HorizontalFootReference {
  vector3_t position;
  vector3_t velocity;
  double weightScale;
};

// World-anchored, time-indexed foot plan shared by the MPC costs and constraints.
// Every query may come from any time inside the planning horizon.
class FootReferencePlan {
 public:
  virtual ~FootReferencePlan() = default;

  virtual const ModeSchedule& modeSchedule() const = 0;
  virtual double positionWeight() const = 0;
  virtual double heading() const = 0;
  // Normalized vertical-load reference for the nominal contact forces.
  virtual std::array<double, 2> supportWeights(double time) const = 0;
  virtual HorizontalFootReference sample(size_t foot, double time) const = 0;
  // When true, sample().position.z()/velocity.z() replace the swing planner's foot height reference.
  virtual bool hasVerticalReference() const { return false; }
};

}  // namespace ocs2::humanoid
