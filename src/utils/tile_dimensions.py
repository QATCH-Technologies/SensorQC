from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def compute_tile_dimensions(z_height):
    # Given data points
    z_values = np.array([6.5, 30.0])
    width_values = np.array([1.4, 6.9])
    height_values = np.array([1.0, 3.8])

    # Fit a power function (log-log transform for better scaling behavior)
    width_coeffs = np.polyfit(np.log(z_values), np.log(width_values), 1)
    height_coeffs = np.polyfit(np.log(z_values), np.log(height_values), 1)

    # Compute predicted width and height
    predicted_width = np.exp(width_coeffs[1]) * z_height**width_coeffs[0]
    predicted_height = np.exp(height_coeffs[1]) * z_height**height_coeffs[0]

    return predicted_width, predicted_height


# Example usage:
z_test = 25
width, height = compute_tile_dimensions(z_test)
print(f"At z-height {z_test}mm, width: {width:.2f}mm, height: {height:.2f}mm")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Define sample z-heights
z_samples = np.linspace(6.5, 100, 10)

cmap = cm.get_cmap('viridis', len(z_samples))
colors = [cmap(i) for i in range(len(z_samples))]
# Draw rectangles for each z-height
for i, z in enumerate(z_samples):
    width, height = compute_tile_dimensions(z)

    # Define rectangle corners at z-height
    x = [-width/2, width/2, width/2, -width/2, -width/2]
    y = [-height/2, -height/2, height/2, height/2, -height/2]
    z_vals = [z] * len(x)

    ax.plot(x, y, z_vals, color=colors[i], linewidth=2)

# Add a solid bottom rectangle (11mm x 10mm) at the lowest Z-height
bottom_z = min(z_samples) - 5  # Slightly below the first tile

x_bottom = [-11/2, 11/2, 11/2, -11/2]
y_bottom = [-10/2, -10/2, 10/2, 10/2]
z_bottom = [bottom_z] * 4

# Create a filled bottom rectangle
verts = [list(zip(x_bottom, y_bottom, z_bottom))]
ax.add_collection3d(Poly3DCollection(verts, color='gray', alpha=0.5))

# Outline the bottom rectangle
ax.plot(x_bottom + [x_bottom[0]], y_bottom + [y_bottom[0]],
        z_bottom + [z_bottom[0]], 'k', linewidth=2)

# Labels and title
ax.set_xlabel('Width (mm)')
ax.set_ylabel('Height (mm)')
ax.set_zlabel('Z-Height (mm)')
ax.set_title('3D Representation')

# Show plot
plt.show()
