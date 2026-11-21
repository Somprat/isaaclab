import torch
from tensordict import TensorClass
from typing import Tuple


def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    unit = x / (x_norm := x.norm(dim=-1, keepdim=True)).clamp(1e-6)
    x = torch.where(x_norm < min, unit * min, x)
    x = torch.where(x_norm > max, unit * max, x)
    return x

def saturate(x: torch.Tensor, a: float):
    norm = x.norm(dim=-1, keepdim=True)
    return (x /norm.clamp_min(1e-6)) * torch.log1p(norm / a) * a

class SpringForce(TensorClass):
    duration: torch.Tensor
    time: torch.Tensor # the time elapsed since the start of the force
    setpoint : torch.Tensor
    setpoint_mass: torch.Tensor
    setpoint_vel: torch.Tensor
    
    kp: torch.Tensor
    kd: torch.Tensor
    
    @classmethod
    def sample(cls, size: int, device: str):
        scalar = torch.empty(size, 1, device=device)
        offset = torch.zeros(size, 3, device=device)
        offset[:, 0] = -0.2
        return cls(
            duration=scalar.uniform_(3., 5.).clone(),
            time=torch.zeros(size, 1, device=device),
            setpoint=offset,
            setpoint_mass=400.*torch.ones(size, 1, device=device),
            setpoint_vel=torch.zeros(size, 3, device=device),
            kp=scalar.uniform_(60., 80.).clone(),
            kd=scalar.uniform_(10., 20.).clone(),
            batch_size=size,
        )
    
    def is_valid(self):
        return self.time < self.duration

    def get_force(self, pos: torch.Tensor, vel: torch.Tensor):
        """Return the world-frame force."""
        force = self.kp * (self.setpoint - pos) - self.kd * vel
        force = clamp_norm(force, 0., 200.) * self.is_valid()
        return force
    
    def step(self, pos: torch.Tensor, vel: torch.Tensor, dt: float):
        self.time.add_(dt)
        self.setpoint_vel.mul_(0.95).add_(-self.get_force(pos, vel) / self.setpoint_mass * dt)
        self.setpoint.add_(self.setpoint_vel * dt)


class ConstantForce(TensorClass):
    """
    A constant pushing/pulling force applied for a certain duration.
    """
    duration: torch.Tensor
    time: torch.Tensor # the time elapsed since the start of the force
    offset: torch.Tensor
    force: torch.Tensor

    @property
    def expired(self):
        return self.time > self.duration - 1e-4

    @classmethod
    def zeros(cls, size: int, device: str):
        return cls(
            duration=torch.zeros(size, 1, device=device),
            time=torch.zeros(size, 1, device=device),
            offset=torch.zeros(size, 3, device=device),
            force=torch.zeros(size, 3, device=device),
            batch_size=size,
        )
    
    @classmethod
    def sample(
        cls,
        size: int,
        duration_range: Tuple[float, float] = (1., 4.),
        force_scales: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        force_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: str = "cpu",
    ):
        duration = torch.zeros(size, 1, device=device)
        duration.uniform_(*duration_range)
        offset = torch.rand(size, 3, device=device) * 2. - 1.
        offset *= torch.as_tensor(force_offsets, device=device)
        force = torch.rand(size, 3, device=device) * 2. - 1.
        force *= torch.as_tensor(force_scales, device=device)
        return cls(
            duration=duration,
            time=torch.zeros(size, 1, device=device),
            offset=offset,
            force=force,
            batch_size=size,
        )

    def get_force(self):
        """Return the world-frame force."""
        return self.force * (self.time < self.duration - 1e-4)


class ImpulseForce(TensorClass):
    """
    A linear impulse force applied for a certain duration with a peak value.
    """
    duration: torch.Tensor
    time: torch.Tensor # the time elapsed since the start of the force
    peak: torch.Tensor

    @property
    def expired(self):
        return self.time > self.duration - 1e-4

    @classmethod
    def zeros(cls, size: int, device: str):
        return cls(
            duration=torch.zeros(size, 1, device=device),
            time=torch.zeros(size, 1, device=device),
            peak=torch.zeros(size, 3, device=device),
            batch_size=size,
        )

    @classmethod
    def sample(
        cls,
        size: int,
        device: str,
        duration_range: Tuple[float, float] = (0.4, 0.6),
        x_range: Tuple[float, float] = (20., 80.),
        y_range: Tuple[float, float] = (20., 80.),
        z_range: Tuple[float, float] = (-20., 20.),
    ):
        duration = torch.zeros(size, 1, device=device)
        duration.uniform_(*duration_range)
        peak = torch.zeros(size, 3, device=device)
        peak[:, 0].uniform_(*x_range)
        peak[:, 1].uniform_(*y_range)
        peak[:, 2].uniform_(*z_range)
        peak *= (torch.rand(size, 3, device=device) - 0.5).sign()
        
        return cls(
            duration=duration,
            time=torch.zeros(size, 1, device=device),
            peak=peak,
            batch_size=size,
        )

    def get_force(self):
        """Return the world-frame force."""
        t = (self.time / self.duration.clamp_min(1e-6)).clamp(0., 1.)
        force = torch.where(t < 0.5, t * 2 * self.peak, (1 - t) * 2 * self.peak)
        return force * (self.time < self.duration - 1e-4)