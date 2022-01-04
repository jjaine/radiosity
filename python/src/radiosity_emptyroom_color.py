import numpy as np
import scipy.sparse.linalg as spla
import scipy.stats
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# For printing, set precision and suppress scientific notation
np.set_printoptions(precision=4, suppress=True)

# Radiosity lighting method for a virtual room, in color.
# The routine "radiosity_emptyroom_Fcomp.py" needs to be computed before
# this one. 

# Adapted from Samuli Siltanen's Matlab code by Essi Jukkala, 2022

# Preliminaries
print("Loading data...")
with open("../data/F_emptyroom", 'rb') as f:
  F = np.load(f)
  n = np.load(f)
  qn = np.load(f)
  d = np.load(f)
  Xmat = np.load(f)
  Ymat = np.load(f)
  Zmat = np.load(f)
print("Data loaded!")

# Adjust the dark shades. Colors darker than the threshold will become
# black, so increasing the threshold will darken the image. 
threshold = 0.05

# Sigmoid correction for optimal gray levels. Increasing betapar1 will
# darken the image, especially the shadows. Increasing betapar2 will
# lighten the image, especially highlights. 
betapar1 = 1
betapar2 = 20

# Camera settings. The camera is located at vector "campos", it is pointed
# towards "camtar", and the view angle is "camang". A larger "camang" value
# will give a more "wide-angle lens", so more of the scene is seen in the
# image. 
campos = [.2, -2.3, -.30]
camtar = [0, 0, 0]
camang = 70

# Construct the color vector (B-vector) using the radiosity lighting model.

# Construct the right hand side Evec of the radiosity equation. Evec
# describes the contribution of emitted light in the scene. For example,
# each pixel belonging to a lamp in the virtual space causes a positive
# element in Evec.
Evec = np.zeros((6 * n**2, 1))
indvec = np.tile(0, len(Evec))
np.power(Xmat[:,1]-0.3, 2)
tempXmat = np.power(Xmat[:,1]-0.3, 2)
tempYMat = np.power(Ymat[:,1], 2)
val = np.sqrt(tempXmat + tempYMat)

# Ceiling lamp
for i in range(0, n**2):
  indvec[n**2 + i] = val[i] < 0.3

for i in range(0, len(indvec)):
  if indvec[i]:
    Evec[i] = 1

print("Right-hand-side constructed")

# The parameter rho adjusts the surface material (how much incoming light
# is reflected away from a patch, 0<rho<=1)
rho = 0.9 * np.ones((6 * n**2, 1))
for i in range(0, n**2):
  rho[n**2 + i] = 1 # Bright ceiling
  rho[2 * n**2 + i] = 0.7; # Dark floor

# Solve for color vector.
print("Solving radiosity equation...")
start = time.time()
colorvec_orig = spla.gmres(np.eye(6 * n**2) - np.tile(rho, [1, 6 * n**2]) * F, Evec)[0]
end = time.time()
print("Radiosity equation solved in", end-start, "seconds")


# Produce a still image of the scene

# Adjust the dark shades and normalize the values of the color vector 
# between 0 and 1.
colorvec = [i - threshold for i in colorvec_orig]
colorvec = [max(0, i) for i in colorvec]
colorvec = colorvec / max(colorvec)

# Sigmoid correction for optimal gray levels.
colorvec = scipy.stats.beta.cdf(colorvec, betapar1, betapar2)


# Construct color matrix 
colorvecR = list(colorvec)
colorvecR[0:n**2] = np.power(colorvecR[0:n**2],0.8) # Back wall
for i in range(0, n**2):
  colorvecR[3 * n**2 + i] = np.power(colorvecR[3 * n**2 + i], 1.2) # Right wall
for i in range(0, n**2):
  colorvecR[4 * n**2 + i] = np.power(colorvecR[4 * n**2 + i], 1.2) # Left wall

colorvecG = list(colorvec)
colorvecG[0:n**2] = np.power(colorvecG[0:n**2], 1.4) # Back wall

colorvecB = list(colorvec)
colorvecB[0:n**2] = np.power(colorvecB[0:n**2], 1.4) # Back wall
for i in range(0, n**2):
  colorvecB[3 * n**2 + i] = np.power(colorvecB[3 * n**2 + i], 0.7) # Right wall
for i in range(0, n**2):
  colorvecB[4 * n**2 + i] = np.power(colorvecB[4 * n**2 + i], 0.7) # Left wall

colormat = [colorvecR[:], colorvecG[:], colorvecB[:]]


# Create plot
fig = plt.figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(projection='3d')

# Draw all the walls consisting of n x n little squares (pixels).
# Pick the gray value of each square from the illumination vector
# calculated by the radiosity method above
colorind = 0

# The back wall
for i in range(0, n**2):
  x = [Xmat[i,0] + d/2, Xmat[i,0] + d/2, Xmat[i,0] - d/2, Xmat[i,0] - d/2]
  y = [Ymat[i,0], Ymat[i,0], Ymat[i,0], Ymat[i,0]]
  z = [Zmat[i,0] - d/2, Zmat[i,0] + d/2, Zmat[i,0] + d/2, Zmat[i,0] - d/2]
  verts = [list(zip(x,y,z))]
  pc = Poly3DCollection(verts)
  color = (colormat[0][colorind], colormat[1][colorind], colormat[2][colorind])
  pc.set_facecolor(color)
  pc.set_edgecolor(color)
  ax.add_collection3d(pc)
  colorind += 1

# Roof
for i in range(0, n**2):
  x = [Xmat[i,1] + d/2, Xmat[i,1] + d/2, Xmat[i,1] - d/2, Xmat[i,1] - d/2]
  y = [Ymat[i,1] - d/2, Ymat[i,1] + d/2, Ymat[i,1] + d/2, Ymat[i,1] - d/2]
  z = [Zmat[i,1], Zmat[i,1], Zmat[i,1], Zmat[i,1]]
  verts = [list(zip(x,y,z))]
  pc = Poly3DCollection(verts)
  color = (colormat[0][colorind], colormat[1][colorind], colormat[2][colorind])
  pc.set_facecolor(color)
  pc.set_edgecolor(color)
  ax.add_collection3d(pc)
  colorind += 1

# Floor
for i in range(0, n**2):
  x = [Xmat[i,2] + d/2, Xmat[i,2] + d/2, Xmat[i,2] - d/2, Xmat[i,2] - d/2]
  y = [Ymat[i,2] - d/2, Ymat[i,2] + d/2, Ymat[i,2] + d/2, Ymat[i,2] - d/2]
  z = [Zmat[i,2], Zmat[i,2], Zmat[i,2], Zmat[i,2]]
  verts = [list(zip(x,y,z))]
  pc = Poly3DCollection(verts)
  color = (colormat[0][colorind], colormat[1][colorind], colormat[2][colorind])
  pc.set_facecolor(color)
  pc.set_edgecolor(color)
  ax.add_collection3d(pc)
  colorind += 1

# Right-hand-side wall
for i in range(0, n**2):
  x = [Xmat[i,3], Xmat[i,3], Xmat[i,3], Xmat[i,3]]
  y = [Ymat[i,3] + d/2, Ymat[i,3] + d/2, Ymat[i,3] - d/2, Ymat[i,3] - d/2]
  z = [Zmat[i,3] - d/2, Zmat[i,3] + d/2, Zmat[i,3] + d/2, Zmat[i,3] - d/2]
  verts = [list(zip(x,y,z))]
  pc = Poly3DCollection(verts)
  color = (colormat[0][colorind], colormat[1][colorind], colormat[2][colorind])
  pc.set_facecolor(color)
  pc.set_edgecolor(color)
  ax.add_collection3d(pc)
  colorind += 1

# Left-hand-side wall
for i in range(0, n**2):
  x = [Xmat[i,4], Xmat[i,4], Xmat[i,4], Xmat[i,4]]
  y = [Ymat[i,4] + d/2, Ymat[i,4] + d/2, Ymat[i,4] - d/2, Ymat[i,4] - d/2]
  z = [Zmat[i,4] - d/2, Zmat[i,4] + d/2, Zmat[i,4] + d/2, Zmat[i,4] - d/2]
  verts = [list(zip(x,y,z))]
  pc = Poly3DCollection(verts)
  color = (colormat[0][colorind], colormat[1][colorind], colormat[2][colorind])
  pc.set_facecolor(color)
  pc.set_edgecolor(color)
  ax.add_collection3d(pc)
  colorind += 1

# Set coordinate limits
plt.xlim([-1.5, 1.5])
plt.ylim([0.5, 1])
ax.set_zlim(-1.5,1)

# TODO: find good values, calculate from campos, camtar & camang?
ax.view_init(elev=1, azim=-89)

plt.axis('off')

plt.show()