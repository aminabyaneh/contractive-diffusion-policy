import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# create a grid of (x, y) values
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
x, y = np.meshgrid(x, y)

# gaussian parameters (σ=2.0 as before)
sigma = 2.0
z = (1 / (2 * np.pi * sigma**2)) * np.exp(- (x**2 + y**2) / (2 * sigma**2))

# convert hex to RGB and create shading based on height
base_color = np.array(mcolors.to_rgb('#5de0e6'))
norm = (z - z.min()) / (z.max() - z.min())
# shade intensity between 0.6 to 1.0 of base color
rgb = base_color * (0.6 + 0.4 * norm[..., None])

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# surface with per-face shading
ax.plot_surface(x, y, z, facecolors=rgb, rstride=1, cstride=1, shade=False, antialiased=True)

# remove grids, axis labels, ticks, and panes for a clean look
ax.grid(False)
ax.set_axis_off()
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# adjust perspective
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()

# grid for density surface
x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
x, y = np.meshgrid(x, y)

# mixture of Gaussians parameters
means = [(-1.5, -1.0), (1.5, 1.0), (0.0, 2.5)]
sigmas = [0.5, 0.7, 0.6]
weights = [0.4, 0.35, 0.25]

# compute mixture density
z = np.zeros_like(x)
for w, mu, s in zip(weights, means, sigmas):
    z += w * (1/(2*np.pi*s**2)) * np.exp(-((x-mu[0])**2 + (y-mu[1])**2)/(2*s**2))

# sample data points from the mixture
n_samples = 500
components = np.random.choice(len(weights), size=n_samples, p=weights)
points = np.zeros((n_samples, 2))
for i, k in enumerate(components):
    points[i, 0] = np.random.normal(loc=means[k][0], scale=sigmas[k])
    points[i, 1] = np.random.normal(loc=means[k][1], scale=sigmas[k])

# prepare RGBA shading for surface
base_color = np.array(mcolors.to_rgb('#004aad'))
norm = (z - z.min())/(z.max() - z.min())
rgb = base_color * (0.6 + 0.4 * norm[..., None])
rgba = np.dstack((rgb, np.full_like(norm, 0.4)))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# density surface
ax.plot_surface(x, y, z, facecolors=rgba, rstride=1, cstride=1, shade=False, antialiased=True)

# scatter raw data at z=0
ax.scatter(points[:, 0], points[:, 1], np.zeros(n_samples),
           color='k', s=10, alpha=0.5)

# clean up for aesthetics
ax.set_axis_off()
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.show()
