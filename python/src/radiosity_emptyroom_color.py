import numpy as np
import scipy.sparse.linalg as spla
import time

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
colorvec_orig = spla.gmres(np.eye(6 * n**2) - np.tile(rho, [1, 6 * n**2]) * F, Evec)
end = time.time()
print("Radiosity equation solved in", end-start, "seconds")
