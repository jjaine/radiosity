import numpy as np
import scipy.sparse.linalg as spla
import scipy.stats
import time
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Finds the index that has the largest unshot radiosity
def get_max_idx(Evec):
    Ecopy = sorted(Evec, reverse=True)
    
    for jj in range(0, len(Ecopy)):
        idx = Evec.index(Ecopy[jj])
        return idx
    
    return -1

# For printing, set precision and suppress scientific notation
np.set_printoptions(precision=4, suppress=True)

# Mosaic resolution
n = 5

# Integration quadrature parameter
qn = 3

# Construct centerpoints of the mosaic tiles (patches),
# d denotes the length of the side of a patch
d = 2/n
tmp = [-1-d/2 + i*d for i in range(1,n+1)]

# Centerpoint coordinate matrices
Xmat = np.zeros((n*n, 6))
Ymat = np.zeros((n*n, 6))
Zmat = np.zeros((n*n, 6))

# Construct the centerpoints for all the tiles in all the six walls

# The back wall (0)
X, Z = np.meshgrid(tmp, tmp)
Xmat[:,0] = np.concatenate(np.transpose(X)).flat
Zmat[:,0] = np.concatenate(np.transpose(Z)).flat
Ymat[:,0] = np.ones(n*n)

# Roof (1)
X, Y = np.meshgrid(tmp, tmp)
Xmat[:,1] = np.concatenate(np.transpose(X)).flat
Ymat[:,1] = np.concatenate(np.transpose(Y)).flat
Zmat[:,1] = np.ones(n*n)

# Floor (2)
Xmat[:,2] = np.concatenate(np.transpose(X)).flat
Ymat[:,2] = np.concatenate(np.transpose(Y)).flat
Zmat[:,2] = -np.ones(n*n)

# Right-hand-side wall (3)
Y, Z = np.meshgrid(tmp, tmp)
Ymat[:,3] = np.concatenate(np.transpose(Y)).flat
Zmat[:,3] = np.concatenate(np.transpose(Z)).flat
Xmat[:,3] = np.ones(n*n)

# Left-hand-side wall (4)
Ymat[:,4] = np.concatenate(np.transpose(Y)).flat
Zmat[:,4] = np.concatenate(np.transpose(Z)).flat
Xmat[:,4] = -np.ones(n*n)

# Front wall (5)
X, Z = np.meshgrid(tmp, tmp)
Xmat[:,5] = np.concatenate(np.transpose(X)).flat
Zmat[:,5] = np.concatenate(np.transpose(Z)).flat
Ymat[:,5] = -np.ones(n*n)

# Adjust the dark shades. Colors darker than the threshold will become
# black, so increasing the threshold will darken the image. 
threshold = 0.005

# Sigmoid correction for optimal gray levels. Increasing betapar1 will
# darken the image, especially the shadows. Increasing betapar2 will
# lighten the image, especially highlights. 
betapar1 = 1
betapar2 = 20

# Construct the color vector (B-vector) using the radiosity lighting model.

# Construct the right hand side Evec of the radiosity equation. Evec
# describes the contribution of emitted light in the scene. For example,
# each pixel belonging to a lamp in the virtual space causes a positive
# element in Evec.
Evec = [0] * (6 * n*n)
indvec = np.tile(0, len(Evec))
tempXmat = np.power(Xmat[:,1]-0.3, 2)
tempYMat = np.power(Ymat[:,1], 2)
val = np.sqrt(tempXmat + tempYMat)

# Ceiling lamp
for i in range(0, n*n):
  indvec[n*n + i] = val[i] < 0.3

# Ceiling lamp for comparison with n=2 case
#for i in range(0, n*n):
    #if i >= 50 and (i % 10) // 5 > 0:
        #indvec[n*n + i] = 1

for i in range(0, len(indvec)):
  if indvec[i]:
    Evec[i] = 1

# Ceiling lamp for the n=2 room
#indvec[7] = 1

for i in range(0, len(indvec)):
  if indvec[i]:
    Evec[i] = 1

print("Right-hand-side constructed")

# The parameter rho adjusts the surface material (how much incoming light
# is reflected away from a patch, 0<rho<=1)
rho = 0.9 * np.ones((6 * n*n, 1))
for i in range(0, n*n):
  rho[n*n + i] = 1 # Bright ceiling
  rho[2 * n*n + i] = 0.7; # Dark floor

# Formula for view factor between square-shaped pixels sharing an edge.
# From Cohen & Wallace: Radiosity and realistic image synthesis
# (Academic Press Professional 1993), Figure 4.4
atan = np.arctan
shared_edge_F = (2 * atan(1) - np.sqrt(2) * atan(1/np.sqrt(2) )+ .25 * np.log(3/4)) / np.pi

# Quadrature points and weights for integrating over a square 
# of size d x d centered at the origin
tt = [i/qn*d - .5*d/qn - d/2 for i in range (1,qn+1)] 
q1, q2 = np.meshgrid(tt, tt)
qw = (d/qn) ** 4 # Area of quadrature pixel, squared, serves as the weight

# Form the geometrical view factor matrix F.
# See http://en.wikipedia.org/wiki/View_factor for details of computation.

# Initialize the matrix
F = np.zeros((6*n*n, 6*n*n))

epsilon = 10 ** -8

# Radiosity vector
B = [x for x in Evec]

start = time.time()

# Start shooting from the patch that has the most unshot radiosity
surfaceIdx = get_max_idx(Evec)
while True:
    ii = surfaceIdx
    for jj in range(0, 6 * n*n):
        # This view factor hasn't been calculated yet
        if F[ii, jj] == 0:
            e1 = ii // (n*n)
            e2 = jj // (n*n)

            if e1 == e2:
                continue

            i = ii % (n*n)
            j = jj % (n*n)

            # Centerpoint of the current pixel
            pi = np.array([Xmat[i, e1], Ymat[i, e1], Zmat[i, e1]])
            # Centerpoint of the other pixel
            pj = np.array([Xmat[j, e2], Ymat[j, e2], Zmat[j, e2]])
            # Distance between the points
            difvec0 = pi-pj
            x = difvec0[0]
            y = difvec0[1]
            z = difvec0[2]
            r0 = math.sqrt(x*x + y*y + z*z)

            # Check if the two pixels share an edge
            if r0 < math.sqrt(2) * d/2 + epsilon: # Edge shared
                # Calculate element of F analytically
                F[ii, jj] = shared_edge_F
                F[jj, ii] = shared_edge_F
            else: # Edge not shared: integrate for F using quadrature
                # Initalize matrix of integrand values at quadrature points
                intgndmat = np.zeros((qn*qn, qn*qn))

                # Roof (1) & floor (2), z constant, indices 0 and 1
                # Left wall (3) & right wall (4), x constant, indices 1 and 2
                # Front (5) & back (0), y constant, indices 0 and 2

                # for current pixel
                i11 = 0
                i12 = 1
                if e1 > 2 and e1 < 5:
                    i11 = 1
                    i12 = 2
                if e1 > 4 or e1 < 1:
                    i12 = 2

                # for other pixel
                i1 = 0
                i2 = 1
                if e2 > 2 and e2 < 5:
                    i1 = 1
                    i2 = 2
                if e2 > 4 or e2 < 1:
                    i2 = 2

                # make normal vectors from qpi and pi
                qpi1 = np.copy(pi)
                qpi1[i11] += q1[1][0]
                qpi1[i12] += q2[1][0]
                qpi2 = np.copy(pi)
                qpi2[i11] += q1[0][1]
                qpi2[i12] += q2[0][1]
                vi1 = pi - qpi1
                vi2 = pi - qpi2

                # make normal vectors from qpj and pj
                qpj1 = np.copy(pj)
                qpj1[i1] += q1[1][0]
                qpj1[i2] += q2[1][0]
                qpj2 = np.copy(pj)
                qpj2[i1] += q1[0][1]
                qpj2[i2] += q2[0][1]
                vj1 = pj - qpj1
                vj2 = pj - qpj2
                ni = np.cross(vi1, vi2)
                nj = np.cross(vj1, vj2)
                x = ni[0]
                y = ni[1]
                z = ni[2]
                ni = ni / math.sqrt(x*x + y*y + z*z)
                x = nj[0]
                y = nj[1]
                z = nj[2]
                nj = nj / math.sqrt(x*x + y*y + z*z)

                # Double loop over four-dimensional quadrature
                for k in range(0, qn*qn):
                    for l in range(0, qn*qn):
                        # Quadrature point in the current pixel
                        qpi = np.copy(pi)
                        qpi[i11] += q1[k%qn][k//qn]
                        qpi[i12] += q2[k%qn][k//qn]

                        # Quadrature point in the other pixel
                        qpj = np.copy(pj)
                        qpj[i1] += q1[l%qn][l//qn]
                        qpj[i2] += q2[l%qn][l//qn]

                        # Vector connecting the quadrature points
                        difvec = qpi - qpj
                        x = difvec[0]
                        y = difvec[1]
                        z = difvec[2]
                        r = math.sqrt(x*x + y*y + z*z)
                        # Calculate the angles
                        cos_i = abs((ni[0] * x + ni[1] * y + ni[2] * z) / r)
                        cos_j = abs((nj[0] * x + nj[1] * y + nj[2] * z) / r)
                        # Evaluate integrand
                        intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)

                # Calculate element of F
                viewfactor = qw * sum(sum(intgndmat)) / d**2
                F[ii, jj] = viewfactor
                F[jj, ii] = viewfactor
        
        # Update radiosity of patch j
        dRad = rho[jj][0] * Evec[ii] * F[ii, jj]
        B[jj] += dRad
        Evec[jj] += dRad

    # Set the unshot radiosity of the current patch to 0
    Evec[ii] = 0
    # Find the next patch to shoot light from
    newIdx = get_max_idx(Evec)
    if newIdx == -1:
        break

    if Evec[newIdx] < 1:
        break
        
    surfaceIdx = newIdx

end = time.time()
print("Radiosity calculated in", end-start, "seconds")

# Check the matrix f, the row sums should all be one
print("Check values: all should ideally be one")
print(sum(F))

# Produce a still image of the scene

# Add ambient term to radiosity
# From Cohen et al: A Progressive Refinement Approach to Fast Radiosity Image Generation
F_approx = 1 / (6 * n*n)
rho_ave = sum(rho)[0] / (6 * n*n)
R = 1 / (1-rho_ave)
ambient = 0
for dB_i in Evec:
    ambient += dB_i * F_approx

ambient *= R

B_amb = B
for i in range(0, len(B)):
    B_amb[i] += B[i] * rho[i][0] + ambient

B = B_amb

# Adjust the dark shades and normalize the values of the color vector 
# between 0 and 1.
colorvec = [i - threshold for i in B]
colorvec = [max(0, i) for i in colorvec]
colorvec = [i / max(colorvec) for i in colorvec]

# Sigmoid correction for optimal gray levels.
colorvec = scipy.stats.beta.cdf(colorvec, betapar1, betapar2)

# Construct color matrix , containing only shades of gray
colormat = [colorvec[:], colorvec[:], colorvec[:]]

# Create plot
fig = plt.figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(projection='3d')

# Draw all the walls consisting of n x n little squares (pixels).
# Pick the gray value of each square from the illumination vector
# calculated by the radiosity method above
colorind = 0

# The back wall
for i in range(0, n*n):
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
for i in range(0, n*n):
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
for i in range(0, n*n):
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
for i in range(0, n*n):
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
for i in range(0, n*n):
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
plt.xlim([-1, 1])
plt.ylim([-1, 1])
ax.set_zlim(-1,1)

# Set the view angle
ax.set_proj_type('persp', 0.25) # Not available in matplotlib < 3.6
ax.view_init(elev=6, azim=-95)

plt.axis('off')

plt.show()