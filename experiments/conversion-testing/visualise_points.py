import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# ---------------------
# Load CSV and Process
# ---------------------
df = pd.read_csv("data/stereo_detections_triangulated.csv").head(25)

# Ensure 'anomaly_detected' is treated as boolean
# df['anomaly_detected'] = df['anomaly_detected'].astype(str) == 'True'

# original_count = len(df)
# df = df[df['anomaly_detected'] == False].copy()
# removed_count = original_count - len(df)

# print(f"Removed {removed_count} anomalous points")

# frame 973 is bounce
# 1011 is return

# offset_to_origin = np.array([-5.7, -3.0, 1.7]) # Apply after converting pixels to metres
offset_to_origin = np.array([10.4, -6.8, 1.7])

# Parse the 3D position strings and convert from mm to meters
trajectory = df['position_3d_pixels'].apply(lambda s: np.array(eval(s)) / 1000.0)
trajectory = np.vstack(trajectory.values)

x = -trajectory[:, 2] + offset_to_origin[0]  # Horizontal position (court width)
y = trajectory[:, 0] + offset_to_origin[1]  # Court length
z = -(trajectory[:, 1]) + offset_to_origin[2]  # Height above court

# ---------------------
# Court Geometry
# ---------------------
court_length = 23.77
court_width_single = 8.23
court_width_double = 10.97
double_alley_width = 1.37
net_post_width = 0.91
net_height = 1.072
net_y = court_length / 2
offset_x = court_width_double / 2
offset_y = court_length / 2


def plot_tennis_court(ax):
    x_plane = np.array([[-offset_x, -offset_x], [offset_x, offset_x]])
    y_plane = np.array([[-offset_y, offset_y], [-offset_y, offset_y]])
    z_plane = np.array([[0, 0], [0, 0]])
    ax.plot_surface(x_plane, y_plane, z_plane, color="green", alpha=0.3)

    # Court lines
    ax.plot([-offset_x, -offset_x], [-offset_y, offset_y], [0, 0], color="black", linewidth=3)
    ax.plot([offset_x, offset_x], [-offset_y, offset_y], [0, 0], color="black", linewidth=3)
    ax.plot([-offset_x, offset_x], [offset_y, offset_y], [0, 0], color="black", linewidth=3)
    ax.plot([-offset_x, offset_x], [-offset_y, -offset_y], [0, 0], color="black", linewidth=3)

    ax.plot([-offset_x + double_alley_width, -offset_x + double_alley_width], [-offset_y, offset_y], [0, 0], color="black", linewidth=2)
    ax.plot([offset_x - double_alley_width, offset_x - double_alley_width], [-offset_y, offset_y], [0, 0], color="black", linewidth=2)

    ax.plot([-offset_x, offset_x], [0, 0], [0, 0], color="black", linestyle="--", linewidth=2)
    ax.plot([-offset_x + net_post_width, offset_x - net_post_width], [0, 0], [net_height, net_height], color="black", linewidth=2)
    ax.plot([-offset_x + net_post_width, -offset_x + net_post_width], [0, 0], [0, net_height], color="black", linewidth=2)
    ax.plot([offset_x - net_post_width, offset_x - net_post_width], [0, 0], [0, net_height], color="black", linewidth=2)

    service_line_y = 6.4
    ax.plot([-offset_x + double_alley_width, offset_x - double_alley_width], [-service_line_y, -service_line_y], [0, 0], color="black", linewidth=2)
    ax.plot([-offset_x + double_alley_width, offset_x - double_alley_width], [service_line_y, service_line_y], [0, 0], color="black", linewidth=2)

    ax.plot([-offset_x + double_alley_width, offset_x - double_alley_width], [-offset_y, -offset_y], [0, 0], color="black", linewidth=2)
    ax.plot([-offset_x + double_alley_width, offset_x - double_alley_width], [offset_y, offset_y], [0, 0], color="black", linewidth=2)

    ax.plot([0, 0], [-service_line_y, service_line_y], [0, 0], color="black", linewidth=2)

    ax.plot([-offset_x + net_post_width, -offset_x + net_post_width], [-0.2, 0.2], [0, 0], color="black", linewidth=4)
    ax.plot([offset_x - net_post_width, offset_x - net_post_width], [-0.2, 0.2], [0, 0], color="black", linewidth=4)

# ---------------------
# Plotting and Slider
# ---------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

plot_tennis_court(ax)

scatter_container = [None]
scatter_container[0] = ax.scatter(x, y, z, c=z, cmap="autumn_r", edgecolor="k", s=50, label="Trajectory Points")
(line,) = ax.plot(x, y, z, color="red", label="Path")

ax.set_zlim(0, 8)
ax.set_xlim(-offset_x, offset_x)
ax.set_ylim(-offset_y, offset_y)
ax.set_aspect('equal')
ax.set_title("3D Tennis Ball Trajectory (Court with Green Surface)")
ax.set_xlabel("Court Width (m)")
ax.set_ylabel("Court Length (m)")
ax.set_zlabel("Height (m)")
ax.legend()

slider_ax = fig.add_axes([0.35, 0.05, 0.3, 0.05])
slider = Slider(ax=slider_ax, label='Frame', color='#aabfaa',
                valmin=1, valmax=len(x), valinit=len(x), valstep=1)
slider.poly.set_edgecolor("black")
slider.poly.set_linewidth(2)

# Display frame number on the right
slider.valtext.set_text(f"{len(x)}/{len(x)}")  # Initialize text

def update(val):
    idx = int(slider.val)
    scatter_container[0].remove()
    new_scatter = ax.scatter(x[:idx], y[:idx], z[:idx],
                             c=z[:idx], cmap="autumn_r",
                             edgecolor="k", s=50)
    line.set_data(x[:idx], y[:idx])
    line.set_3d_properties(z[:idx])
    scatter_container[0] = new_scatter
    slider.valtext.set_text(f"{idx}/{len(x)}")  # Update frame count
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
