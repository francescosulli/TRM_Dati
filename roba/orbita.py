import numpy as np
import matplotlib.pyplot as plt

# Define orbital parameters
semi_major_axis = 7000  # km (semi-major axis)
eccentricity = 0.001    # Almost circular orbit
theta = np.linspace(0, 2*np.pi, 500)  # True anomalies

# Calculate the position of the orbit (polar coordinates)
r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(theta))  # Orbital radius at each angle

# Convert to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)

# Plot the orbit
fig, ax = plt.subplots(figsize=(8, 8))

# Plot Earth at the center
ax.plot(0, 0, 'go', markersize=10, label="Earth")

# Plot the orbit trajectory
ax.plot(x, y, label="Orbit", color="blue")

# Set labels and title
ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_title("Satellite Orbit Around Earth")
ax.legend()

# Set equal scaling and grid
ax.set_aspect('equal')
ax.grid(True)

# Show the plot
plt.show()