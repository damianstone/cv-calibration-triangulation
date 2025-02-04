# https://www.researchgate.net/figure/mpact-height-ball-trajectory-angle-and-ball-speed-at-impact-position_fig10_322532284
# https://link.springer.com/article/10.1007/s12283-013-0144-9
# http://spiff.rit.edu/richmond/baseball/traj/traj.html
# https://www.quintic.com/education/case_studies/coefficient_restitution.htm#:~:text=The%20coefficients%20of%20restitution%20vary,energy%20was%20lost%20during%20impact
# https://isjos.org/ISBJOP/JoPv1i1-2Tennis.pdf

import numpy as np
import matplotlib.pyplot as plt

# Constants
gravity = 9.81  # acceleration due to gravity (m/s^2)
mass = 0.0594  # mass of a tennis ball (kg)
radius = 0.0335  # radius of the ball (m)
dt = 1 / 60  # time interval between frames (s)
time_horizon = 3.0 # prediction duration (s)
drag_coeff = 0.507  # drag coefficient
air_density = 1.204  # air density (kg/m^3)
cross_sectional_area = 0.00456  # cross-sectional area of the ball (m^2)
spin_vector = np.array([0, 0, -1])  # spin vector of the ball
num_predictions = int(time_horizon / dt)  # number of prediction points

def restitution_coeff(v0, coeff_variance=0.07, exp_variance=0.1):
    """Calculate the coefficient of restitution based on initial velocity."""
    coefficient = 0.18 + np.random.uniform(-coeff_variance, coeff_variance)
    exponent = 0.5 + np.random.uniform(-exp_variance, exp_variance)
    return max(1 - coefficient * v0 ** exponent, 0)

def predict_trajectory(data_points):
    """Predict the trajectory of a tennis ball based on initial data points."""
    trajectory = data_points.copy()

    x_coords, y_coords, z_coords = zip(*data_points)
    x, y, z = data_points[-1]

    vx, vy, vz = (np.diff(coords)[-1] / dt for coords in (x_coords, y_coords, z_coords))

    for _ in range(num_predictions):
        if z <= 0 and vz <= 0:
            vz *= -restitution_coeff(abs(vz))
            if abs(vz) < 0.1:
                break

        # Velocity magnitude
        v = max(np.sqrt(vx**2 + vy**2 + vz**2), 1e-6)

        # Drag force
        drag_force = 0.5 * drag_coeff * air_density * cross_sectional_area * v**2
        drag_acc = drag_force / mass
        drag_vector = drag_acc * np.array([vx, vy, vz]) / v

        # Magnus (lift) force
        magnus_force = (
            0.5 * air_density * cross_sectional_area * v**2 *
            np.cross(spin_vector, [vx, vy, vz]) / np.linalg.norm([vx, vy, vz])
        )
        magnus_acc = magnus_force / mass

        # Accelerations
        ax = -drag_vector[0] + magnus_acc[0]
        ay = -drag_vector[1] + magnus_acc[1]
        az = -gravity - drag_vector[2] + magnus_acc[2]

        # Update velocities
        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        # Update positions
        x += vx * dt
        y += vy * dt
        z += vz * dt

        trajectory.append((x, y, z))

    return trajectory

# Initial data points
data_points = [
    (3.0, -11.5, 1.0),
    (2.88, -11.25, 1.06),
    (2.77, -11, 1.13),
    (2.65, -10.75, 1.2)
]

# Predict trajectory
predicted_trajectory = predict_trajectory(data_points)

# Save predictions to a file
with open("prediction.txt", "w") as file:
    for point in predicted_trajectory:
        file.write(f"{tuple(map(float, point))},\n")

# Plot the trajectory
z_coords = [point[2] for point in predicted_trajectory]
times = np.linspace(0, len(predicted_trajectory) * dt, len(predicted_trajectory))

plt.plot(times, z_coords, label="Z Coordinate")
plt.title("Tennis Ball Trajectory (Z over Time)")
plt.xlabel("Time (s)")
plt.ylabel("Height (z)")
plt.legend()
plt.grid()
plt.show()
