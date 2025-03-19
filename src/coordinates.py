import numpy as np
import matplotlib.pyplot as plt


def simulate_trajectory(
    x0,
    y0,
    z0,
    vx0,
    vy0,
    vz0,
    mass,
    diameter,
    drag_coeff,
    lift_coeff,
    spin_factor,
    air_density,
    time_step,
    frame_rate,
):

    # Constants
    g = 9.8  # Gravitational acceleration (m/s^2)
    radius = diameter / 2
    cross_sectional_area = np.pi * radius**2  # Cross-sectional area of the ball

    # Initial conditions
    x, y, z = x0, y0, z0
    vx, vy, vz = vx0, vy0, vz0

    # Trajectory storage
    trajectory = [(x, y, z)]

    # Frame capture time step based on frame rate
    frame_interval = 1 / frame_rate
    time_elapsed = 0

    # Simulation loop
    while z > 0:
        # Calculate velocity magnitude
        v = np.sqrt(vx**2 + vy**2 + vz**2)

        # Model expenontially decreasing spin
        lift_coeff *= np.exp(-1 * time_step)

        # Calculate drag force magnitude
        drag = 0.5 * air_density * drag_coeff * cross_sectional_area * v**2
        magnus_force = 0.5 * air_density * lift_coeff * cross_sectional_area * v**2

        # Calculate accelerations
        ax = (
            (-(drag * vx) + (magnus_force * -vz * spin_factor[0])) / (mass * v)
            if v != 0
            else 0
        )
        ay = (
            (-(drag * vy) + (magnus_force * vx * spin_factor[1])) / (mass * v)
            if v != 0
            else 0
        )
        az = (
            -g - ((drag * vz) - (magnus_force * vy * spin_factor[2])) / (mass * v)
            if v != 0
            else -g
        )

        # Update velocities
        vx += ax * time_step
        vy += ay * time_step
        vz += az * time_step

        # Update positions
        x += vx * time_step
        y += vy * time_step
        z += vz * time_step

        # Update time
        time_elapsed += time_step

        # Capture the frame if the time elapsed has reached the frame interval
        if time_elapsed >= frame_interval:
            trajectory.append((x, y, z))
            time_elapsed = 0  # Reset time counter for next frame

        # Stop if the ball hits the ground
        if z <= 0:
            break

    return trajectory


# Parameters
x0, y0, z0 = 0, 3, 1.0  # Initial position (meters)
vx0, vy0, vz0 = 15, -7, 4  # Initial velocity components (m/s)
mass = 0.058  # Mass of a tennis ball (kg)
diameter = 0.067  # Diameter of a tennis ball (meters)
drag_coeff = 0.5  # Drag coefficient (unitless)
lift_coeff = 0.8  # Approximated experimentally
air_density = 1.225  # Air density at sea level (kg/m^3)
time_step = 0.025  # Time step (seconds)
frame_rate = 60  # Camera frame rate (frames per second)
spin_factor = (
    0,
    0,
    -1,
)  # Simulate different spin orientations (0.1, 0.35, 0.55) = base top spin
court_length = 23.77
doubles_width = 10.97
net_height = 0.914
service_box_length = 6.4  # meters
service_box_width = 4.115  # meters (half the width of the doubles court)
tramline_width = 1.37  # Width of the tramlines (meters)

# Simulate trajectory
trajectory = simulate_trajectory(
    x0,
    y0,
    z0,
    vx0,
    vy0,
    vz0,
    mass,
    diameter,
    drag_coeff,
    lift_coeff,
    spin_factor,
    air_density,
    time_step,
    frame_rate,
)

# Extract coordinates
trajectory = np.array(trajectory)

# Add random error of up radius 5.5 / 2 cm in each direction (x, y, z) directly to the trajectory array
error_margin = 0.055 / 2

# Generate random noise for each coordinate
random_error = np.random.uniform(-error_margin, error_margin, trajectory.shape)
trajectory = trajectory + random_error
x_vals, y_vals, z_vals = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

# Plot trajectory with tennis court
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot the tennis ball points
ax.scatter(
    x_vals, y_vals, z_vals, marker="o", color="blue", label="Tennis Ball Trajectory"
)

# Court dimensions (length x width)
x_court = np.linspace(0, court_length, 2)
y_court = np.linspace(-doubles_width / 2, doubles_width / 2, 2)
X, Y = np.meshgrid(x_court, y_court)  # Create a grid for the court surface
Z = np.zeros_like(X)  # The court surface is at z = 0

# Plot the court surface
ax.plot_surface(X, Y, Z, color="green", alpha=0.5)

# Tramlines
ax.plot(
    [0, court_length],
    [doubles_width / 2 - tramline_width, doubles_width / 2 - tramline_width],
    [0, 0],
    color="white",
    linewidth=2,
)  # Left tramline
ax.plot(
    [0, court_length],
    [-doubles_width / 2 + tramline_width, -doubles_width / 2 + tramline_width],
    [0, 0],
    color="white",
    linewidth=2,
)  # Right tramline

# Center line
ax.plot(
    [court_length / 2 - service_box_length, court_length / 2 + service_box_length],
    [0, 0],
    [0, 0],
    color="white",
    linewidth=2,
)

# Service boxes
ax.plot(
    [court_length / 2 - service_box_length, court_length / 2 - service_box_length],
    [doubles_width / 2 - tramline_width, -doubles_width / 2 + tramline_width],
    [0, 0],
    color="white",
    linewidth=2,
)
ax.plot(
    [court_length / 2 + service_box_length, court_length / 2 + service_box_length],
    [doubles_width / 2 - tramline_width, -doubles_width / 2 + tramline_width],
    [0, 0],
    color="white",
    linewidth=2,
)

# Netline
ax.plot(
    [court_length / 2, court_length / 2],
    [-doubles_width / 2, doubles_width / 2],
    [net_height, net_height],
    color="red",
    linewidth=2,
    label="Net",
)

# Add the net
net_x = [court_length / 2, court_length / 2]
net_y = [-doubles_width / 2, doubles_width / 2]
net_z = [0, net_height]
for y in net_y:
    ax.plot(
        [net_x[0], net_x[0]],
        [y, y],
        [0, net_height],
        color="red",
        label="Net" if y == net_y[0] else "",
    )

# Restrict plot limits to court dimensions
ax.set_xlim([0, court_length])
ax.set_ylim([-doubles_width / 2, doubles_width / 2])
ax.set_zlim([0, 2])  # Set a reasonable height limit for better visualization
ax.set_box_aspect([court_length, doubles_width, 4])  # Scale X, Y, Z

# Labels and legend
ax.grid(False)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("Tennis Ball Trajectory with Court Overlay")
ax.legend()
plt.show()
