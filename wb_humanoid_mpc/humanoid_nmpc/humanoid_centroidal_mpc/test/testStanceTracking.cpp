#include <gtest/gtest.h>
#include <humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

using namespace ocs2;
using namespace ocs2::humanoid;

// A translating, yawing foot with input equal to world twist. This isolates
// the heading feedback sign and its Jacobian from the robot's kinematics.
class YawFoot final : public EndEffectorKinematics<scalar_t> {
 public:
  YawFoot* clone() const override { return new YawFoot(*this); }
  const std::vector<std::string>& getIds() const override { return ids_; }
  std::vector<vector3_t> getPosition(const vector_t& x) const override { return {x.head<3>()}; }
  std::vector<quaternion_t> getOrientation(const vector_t& x) const override {
    return {quaternion_t(Eigen::AngleAxisd(x(3), vector3_t::UnitZ()))};
  }
  std::vector<vector3_t> getOrientationError(const vector_t& x, const std::vector<quaternion_t>& r) const override {
    return {quaternionDistance(getOrientation(x).front(),r.front())};
  }
  std::vector<vector3_t> getOrientationErrorWrtPlane(const vector_t&,const std::vector<vector3_t>&) const override {
    return {vector3_t::Zero()};
  }
  std::vector<vector6_t> getTwist(const vector_t&,const vector_t& u) const override { return {u}; }
  std::vector<VectorFunctionLinearApproximation> getPositionLinearApproximation(const vector_t& x) const override {
    auto a=VectorFunctionLinearApproximation::Zero(3,x.size(),0);
    a.f=x.head<3>(); a.dfdx.leftCols(3).setIdentity(); return {a};
  }
  std::vector<VectorFunctionLinearApproximation> getOrientationErrorLinearApproximation(
      const vector_t& x,const std::vector<quaternion_t>& r) const override {
    auto a=VectorFunctionLinearApproximation::Zero(3,x.size(),0);
    const auto yaw=2*std::atan2(r.front().z(),r.front().w());
    a.f=getOrientationError(x,r).front(); a.dfdx(2,3)=-.5*std::cos((x(3)-yaw)/2); return {a};
  }
  std::vector<VectorFunctionLinearApproximation> getOrientationErrorWrtPlaneLinearApproximation(
      const vector_t& x,const std::vector<vector3_t>&) const override {
    return {VectorFunctionLinearApproximation::Zero(3,x.size(),0)};
  }
  std::vector<VectorFunctionLinearApproximation> getTwistLinearApproximation(const vector_t& x,const vector_t& u) const override {
    auto a=VectorFunctionLinearApproximation::Zero(6,x.size(),u.size()); a.f=u; a.dfdu.setIdentity(); return {a};
  }
  std::vector<vector3_t> getVelocity(const vector_t&,const vector_t&) const override { throw std::logic_error("unused"); }
  std::vector<vector3_t> getAngularVelocity(const vector_t&,const vector_t&) const override { throw std::logic_error("unused"); }
  std::vector<VectorFunctionLinearApproximation> getVelocityLinearApproximation(const vector_t&,const vector_t&) const override {
    throw std::logic_error("unused");
  }
  std::vector<VectorFunctionLinearApproximation> getAngularVelocityLinearApproximation(const vector_t&,const vector_t&) const override {
    throw std::logic_error("unused");
  }
 private:
  const std::vector<std::string> ids_{"foot"};
};

TEST(StanceTracking, HeadingFeedbackAndJacobian) {
  YawFoot foot;
  EndEffectorKinematicsTwistConstraint::Config config;
  config.Ax=5*matrix_t::Identity(6,6); config.Av=matrix_t::Identity(6,6); config.b=vector_t::Zero(6);
  vector_t x(4); x << .01,-.02,.03,.3;
  vector_t u=vector_t::Zero(6); PreComputation preComp;
  for (bool heading : {false,true}) {
    if (heading) config.orientationReference=Eigen::Quaterniond(Eigen::AngleAxisd(.1,vector3_t::UnitZ()));
    EndEffectorKinematicsTwistConstraint constraint(foot,6,config);
    const auto a=constraint.getLinearApproximation(0,x,u,preComp);
    EXPECT_TRUE(a.f.isApprox(constraint.getValue(0,x,u,preComp),1e-12));
    for (int j=0;j<x.size();++j) {
      auto shifted=x; shifted(j)+=1e-6;
      const vector_t diff=(constraint.getValue(0,shifted,u,preComp)-a.f)/1e-6;
      EXPECT_LT((diff-a.dfdx.col(j)).norm(),1e-6);
    }
    for (int j=0;j<u.size();++j) {
      auto shifted=u; shifted(j)+=1e-6;
      const vector_t diff=(constraint.getValue(0,x,shifted,preComp)-a.f)/1e-6;
      EXPECT_LT((diff-a.dfdu.col(j)).norm(),1e-6);
    }
    if (heading) EXPECT_GT(a.f(5),0);  // Constraint demands negative yaw rate toward reference.
    else EXPECT_DOUBLE_EQ(a.f(5),0);  // Legacy plane alignment leaves yaw free.
  }
}
