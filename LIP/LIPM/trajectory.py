"""Reusable LIPM startup and swing generation; coordinates are world Z-up.

Startup is a constant-height, two-foot-support LIPM boundary-value problem.
Its ZMP/friction checks concern the reduced model, not whole-body feasibility.
"""

import numpy as np
from LIPM_3D import LIPM3D
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def smooth_swing(start, target, u, duration, height):
    """Foot position/velocity/acceleration with zero derivatives at both ends."""
    u = np.asarray(u)[..., None]
    delta = np.asarray(target) - start
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
    rate = 30 * u**2 * (1 - u) ** 2 / duration
    accel = 60 * u * (1 - u) * (1 - 2 * u) / duration**2
    position = np.asarray(start) + blend * delta
    velocity = rate * delta
    acceleration = accel * delta
    # Sixth-order arch: unit apex and zero velocity AND acceleration at both ends.
    position[..., 2] += (64 * height * u**3 * (1 - u) ** 3)[..., 0]
    velocity[..., 2] += (192 * height * u**2 * (1 - u) ** 2 * (1 - 2 * u) / duration)[..., 0]
    acceleration[..., 2] += (384 * height * u * (1 - u) * (1 - 5 * u + 5 * u**2) / duration**2)[..., 0]
    return position, velocity, acceleration


def polynomial_state(coefficients, u, duration, derivative=0):
    """Evaluate normalized-time coefficients and analytical time derivatives."""
    coefficients = np.polynomial.polynomial.polyder(coefficients, derivative, axis=0) / duration**derivative
    return np.polynomial.polynomial.polyval(np.asarray(u), coefficients).T


def _extrema(coefficients):
    roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients))
    times = [
        0.0,
        1.0,
        *[r.real for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < 1],
    ]
    return np.polynomial.polynomial.polyval(times, coefficients)


def plan_startup(
    com,
    velocity,
    feet,
    step_length,
    step_width,
    step_duration,
    heading,
    dt,
    duration=0.8,
    max_duration=2.0,
    contact_points=None,
):
    """Minimize CoM jerk, with terminal capture state fixing the first landing.

    A feasible endpoint is chosen automatically. Increase duration if needed.
    Start acceleration is zero; end acceleration matches left-support LIPM.
    The ZMP stays 3 mm inside the convex support hull. Acceleration is bounded
    by mu*g with mu=0.35 (below the preset's 0.4 friction). Polynomial extrema
    validate the entire startup interval after the sampled optimizer constraints.
    """
    w = np.sqrt(9.81 / com[2])
    rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    offset = np.array(
        [
            step_length / np.expm1(w * step_duration),
            -step_width / (np.exp(w * step_duration) + 1),
        ]
    )
    capture = feet[0, :2] + rotation @ offset
    if contact_points is None:
        # Matched G1 URDF contact-sphere centers projected onto the ground.
        sole_corners = np.array([[-0.05, -0.025], [-0.05, 0.025], [0.12, -0.03], [0.12, 0.03]])
        contact_points = np.concatenate([foot[:2] + sole_corners @ rotation.T for foot in feet])
    hull = ConvexHull(np.asarray(contact_points)[..., :2].reshape(-1, 2))
    normals, bounds = hull.equations[:, :2], hull.equations[:, 2]
    u = np.linspace(0, 1, 201)
    matrix = np.array([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [6.0, 12.0, 20.0]])
    first = int(np.ceil(duration / dt - 1e-9))
    last = int(np.floor(max_duration / dt + 1e-9))
    increment = max(1, round(0.2 / dt))
    candidates = sorted({*range(first, last + 1, increment), last})
    for steps in candidates:
        elapsed = steps * dt

        def coefficients(end_position, elapsed=elapsed):
            c = np.zeros((6, 2))
            c[0] = com[:2]
            c[1] = velocity[:2] * elapsed
            end_velocity = w * (capture - end_position)
            end_acceleration = w**2 * (end_position - feet[0, :2])
            c[3:] = np.linalg.solve(
                matrix,
                np.stack(
                    (
                        end_position - c[0] - c[1],
                        end_velocity * elapsed - c[1],
                        end_acceleration * elapsed**2,
                    )
                ),
            )
            return c

        def constraints(end_position, elapsed=elapsed, coefficients=coefficients):
            c = coefficients(end_position)
            p = polynomial_state(c, u, elapsed)
            a = polynomial_state(c, u, elapsed, 2)
            return np.r_[
                -((p - a / w**2) @ normals.T + bounds).ravel() - 0.003,
                (0.35 * 9.81) ** 2 - (a * a).sum(-1),
            ]

        result = minimize(
            lambda p, elapsed=elapsed, coefficients=coefficients: np.mean(
                polynomial_state(coefficients(p), u, elapsed, 3) ** 2
            ),
            com[:2],
            method="SLSQP",
            constraints=[{"type": "ineq", "fun": constraints}],
            options={"ftol": 1e-10, "maxiter": 250},
        )
        if not result.success or constraints(result.x).min() < -1e-7:
            continue
        c = coefficients(result.x)
        acc_coefficients = np.polynomial.polynomial.polyder(c, 2, axis=0) / elapsed**2
        zmp_coefficients = c.copy()
        zmp_coefficients[:4] -= acc_coefficients / w**2
        margins = []
        for normal, bound in zip(normals, bounds):
            residual = zmp_coefficients @ normal
            residual[0] += bound
            margins.append(-max(_extrema(residual)))
        accel_squared = sum(
            np.polynomial.polynomial.polymul(acc_coefficients[:, j], acc_coefficients[:, j]) for j in range(2)
        )
        max_acceleration = np.sqrt(max(0.0, max(_extrema(accel_squared))))
        if min(margins) < 0.003 - 1e-7 or max_acceleration > 0.35 * 9.81 + 1e-7:
            continue
        return c, elapsed, min(margins), max_acceleration, hull.points[hull.vertices]
    raise ValueError(
        "No feasible two-foot LIPM startup within the requested durations. "
        "Check initial velocity/CoM, foot geometry and gait commands; do not export an unchecked clip."
    )


def validate_schedule(schedule, duration, transition_steps):
    """Validate time/length/duration rows; times include the startup phase."""
    rows = np.asarray(schedule, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3 or len(rows) == 0:
        raise ValueError("schedule must contain [time, step_length, step_duration] rows")
    if (
        not np.isfinite(rows).all()
        or rows[0, 0] != 0
        or not (np.diff(rows[:, 0]) > 0).all()
        or rows[-1, 0] >= duration
        or (rows[:, 1] < 0).any()
        or (rows[:, 2] < 0.04).any()
    ):
        raise ValueError("schedule needs increasing times from zero, nonnegative lengths and durations >= 0.04")
    if isinstance(transition_steps, bool) or int(transition_steps) != transition_steps or transition_steps < 1:
        raise ValueError("transition_steps must be a positive integer")
    return rows


def generate_reference(
    com,
    velocity,
    feet,
    *,
    step_length,
    step_width,
    step_duration,
    heading=0.0,
    swing_height=0.05,
    duration=10.0,
    dt=0.02,
    startup_duration=0.8,
    max_startup_duration=2.0,
    contact_points=None,
    schedule=None,
    transition_steps=2,
):
    """Generate schema-2 arrays, including analytic velocities and a startup phase."""
    scheduled = schedule is not None
    if scheduled:
        schedule = validate_schedule(schedule, duration, transition_steps)
        step_length, step_duration = schedule[0, 1:]
    com, velocity, feet = (
        np.asarray(com, dtype=float),
        np.asarray(velocity, dtype=float),
        np.asarray(feet, dtype=float).copy(),
    )
    if com.shape != (3,) or velocity.shape != (2,) or feet.shape != (2, 3):
        raise ValueError("Expected CoM (3,), horizontal velocity (2,), feet (2,3)")
    scalars = np.array(
        [
            step_length,
            step_width,
            step_duration,
            heading,
            swing_height,
            duration,
            dt,
            startup_duration,
            max_startup_duration,
        ]
    )
    if not all(np.isfinite(x).all() for x in (com, velocity, feet, scalars)):
        raise ValueError("Inputs must be finite")
    if (
        com[2] <= 0
        or min(step_width, step_duration, duration, dt) <= 0
        or min(step_length, swing_height, startup_duration) < 0
    ):
        raise ValueError("Invalid height, gait, timestep or duration")
    if max_startup_duration < startup_duration:
        raise ValueError("max_startup_duration must be at least startup_duration")
    feet[:, 2] = 0
    steps_per_swing = max(2, round(step_duration / dt))
    if not scheduled:
        dt = step_duration / steps_per_swing
    else:
        step_duration = steps_per_swing * dt
    w = np.sqrt(9.81 / com[2])
    coefficients = None
    actual_startup = margin = max_acceleration = 0.0
    support_polygon = np.empty((0, 2))
    if startup_duration > 0:
        coefficients, actual_startup, margin, max_acceleration, support_polygon = plan_startup(
            com,
            velocity,
            feet,
            step_length,
            step_width,
            step_duration,
            heading,
            dt,
            startup_duration,
            max_startup_duration,
            contact_points,
        )
    if duration < actual_startup + step_duration:
        raise ValueError("Rollout must include startup and at least one complete step")
    startup_steps = round(actual_startup / dt)
    time = np.arange(int(np.floor(duration / dt + 1e-9)) + 1) * dt
    n = len(time)
    positions = np.tile(com, (n, 1))
    velocities = np.zeros((n, 3))
    accelerations = np.zeros((n, 3))
    foot_positions = np.tile(feet, (n, 1, 1))
    foot_velocities = np.zeros_like(foot_positions)
    foot_accelerations = np.zeros_like(foot_positions)
    support = np.zeros((n, 2), dtype=bool)
    phase = np.zeros(n)
    zmp = np.zeros((n, 3))
    end_com, end_velocity = com.copy(), velocity.copy()
    if coefficients is not None:
        u = time[:startup_steps] / actual_startup
        positions[:startup_steps, :2] = polynomial_state(coefficients, u, actual_startup)
        velocities[:startup_steps, :2] = polynomial_state(coefficients, u, actual_startup, 1)
        accelerations[:startup_steps, :2] = polynomial_state(coefficients, u, actual_startup, 2)
        zmp[:startup_steps, :2] = positions[:startup_steps, :2] - accelerations[:startup_steps, :2] / w**2
        support[:startup_steps] = True
        end_com[:2] = polynomial_state(coefficients, 1.0, actual_startup)
        end_velocity = polynomial_state(coefficients, 1.0, actual_startup, 1)
    model = LIPM3D(dt=dt, T=step_duration, T_d=step_duration, s_d=step_length, w_d=step_width)
    model.initializeModel(end_com.tolist(), feet[0].tolist(), feet[1].tolist())
    model.x_0, model.y_0 = end_com[:2] - feet[0, :2]
    model.vx_0, model.vy_0 = end_velocity
    model.x_t, model.y_t, model.vx_t, model.vy_t = (
        model.x_0,
        model.y_0,
        model.vx_0,
        model.vy_0,
    )
    lengths = np.full(n, step_length)
    periods = np.full(n, step_duration)
    applied = [[actual_startup, step_length, step_duration]]
    schedule_index = 0
    blend_index = transition_steps
    blend_start = np.array([step_length, step_duration])
    blend_target = blend_start.copy()
    swing_id = 1
    model.calculateFootLocationForNextStepXcoMWorld(heading)
    target = np.array([model.u_x, model.u_y, 0.0])
    first_landing = target.copy()
    start = feet[swing_id].copy()
    step_start = startup_steps
    completed_steps = 0
    for i in range(startup_steps, n):
        k = i - startup_steps
        if k:
            model.step()
            fraction = (i - step_start) / steps_per_swing
            (
                feet[swing_id],
                foot_velocities[i, swing_id],
                foot_accelerations[i, swing_id],
            ) = smooth_swing(start, target, fraction, step_duration, swing_height)
            if swing_id == 1:
                model.right_foot_pos = feet[1].tolist()
            else:
                model.left_foot_pos = feet[0].tolist()
            if i - step_start == steps_per_swing:
                model.switchSupportLeg()
                model.x_t, model.y_t = model.x_0, model.y_0
                swing_id = 1 - swing_id
                start = feet[swing_id].copy()
                step_start = i
                completed_steps += 1
                if scheduled:
                    index = np.searchsorted(schedule[:, 0], time[i] + 1e-9, side="right") - 1
                    if index != schedule_index:
                        schedule_index = index
                        blend_start = np.array([step_length, step_duration])
                        blend_target = schedule[index, 1:]
                        blend_index = 0
                    if blend_index < transition_steps:
                        blend_index += 1
                        step_length, requested_period = (
                            blend_start + (blend_target - blend_start) * blend_index / transition_steps
                        )
                        steps_per_swing = max(2, round(requested_period / dt))
                        step_duration = steps_per_swing * dt
                        model.s_d = step_length
                        model.T = model.T_d = step_duration
                        applied.append([time[i], step_length, step_duration])
                model.calculateFootLocationForNextStepXcoMWorld(heading)
                target = np.array([model.u_x, model.u_y, 0.0])
        positions[i, :2] = np.array([model.x_t, model.y_t]) + model.support_foot_pos[:2]
        velocities[i, :2] = model.vx_t, model.vy_t
        accelerations[i, :2] = w**2 * np.array([model.x_t, model.y_t])
        foot_positions[i] = feet
        support[i, 1 - swing_id] = True
        phase[i] = ((completed_steps + (i - step_start) / steps_per_swing) / 2) % 1
        lengths[i], periods[i] = step_length, step_duration
        zmp[i, :2] = model.support_foot_pos[:2]
    return {
        "schema_version": np.array(3 if scheduled else 2),
        "frame": np.array("world_z_up"),
        "foot_names": np.array(["left", "right"]),
        "time": time,
        "dt": np.array(dt),
        "com_pos": positions,
        "com_vel": velocities,
        "com_acc": accelerations,
        "foot_pos": foot_positions,
        "foot_vel": foot_velocities,
        "foot_acc": foot_accelerations,
        "support_mask": support,
        "phase": phase,
        "zmp": zmp,
        "desired_step_length": lengths,
        "desired_step_width": np.full(n, step_width),
        "step_duration": periods if scheduled else np.array(step_duration),
        "heading": np.array(heading),
        "com_height": np.array(com[2]),
        "swing_height": np.array(swing_height),
        "initial_velocity": velocity,
        "startup_duration": np.array(actual_startup),
        "startup_zmp_margin": np.array(margin),
        "startup_max_acceleration": np.array(max_acceleration),
        "startup_support_polygon": support_polygon,
        "startup_coefficients": np.empty((0, 2)) if coefficients is None else coefficients,
        "first_landing": first_landing,
        "gait_schedule": schedule if scheduled else np.empty((0, 3)),
        "applied_gait_changes": np.asarray(applied),
        "transition_steps": np.array(transition_steps if scheduled else 0),
    }
