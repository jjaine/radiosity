import numpy as np
import time
import math

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

start = time.time()

# From the roof (j) to the back wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the back wall
    pi = np.array([Xmat[i, 0], Ymat[i, 0], Zmat[i, 0]])
    # Centerpoint of the current pixel in the roof
    pj = np.array([Xmat[j, 1], Ymat[j, 1], Zmat[j, 1]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < math.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[i, n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the back wall pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1], pi[2] + q2[k%qn][k//qn]])
          # Quadrature point in the roof pixel
          qpj = np.array([pj[0] + q1[l%qn][l//qn], pj[1] + q2[l%qn][l//qn], pj[2]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[1] / r)
          cos_j = abs(difvec[2] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)

      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[i, n*n + j] = viewfactor

print("Geometric view factors roof->back done (1/15)")

# From the floor (j) to the back wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the back wall
    pi =  np.array([Xmat[i, 0], Ymat[i, 0], Zmat[i, 0]])
    # Centerpoint of the current pixel in the floor
    pj =  np.array([Xmat[j, 2], Ymat[j, 2], Zmat[j, 2]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[i, 2 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the back wall pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1], pi[2] + q2[k%qn][k//qn]])
          # Quadrature point in the floor pixel
          qpj = np.array([pj[0] + q1[l%qn][l//qn], pj[1] + q2[l%qn][l//qn], pj[2]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[2] / r)
          cos_j = abs(difvec[1] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)

      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[i, 2 * n*n + j] = viewfactor

print("Geometric view factors floor->back done (2/15)")

# From the right-hand-side wall (j) to the back wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the back wall
    pi = np.array([Xmat[i, 0], Ymat[i, 0], Zmat[i, 0]])
    # Centerpoint of the current pixel in the right-hand-side wall
    pj = np.array([Xmat[j, 3], Ymat[j, 3], Zmat[j, 3]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[i, 3 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the back wall pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1], pi[2] + q2[k%qn][k//qn]])
          # Quadrature point in the right wall pixel
          qpj = np.array([pj[0], pj[1] + q1[l%qn][l//qn], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[0] / r)
          cos_j = abs(difvec[1] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[i, 3 * n*n + j] = viewfactor  

print("Geometric view factors right->back done (3/15)")

# From the left-hand-side wall (j) to the back wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the back wall
    pi = np.array([Xmat[i, 0], Ymat[i, 0], Zmat[i, 0]])
    # Centerpoint of the current pixel in the left-hand-side wall
    pj = np.array([Xmat[j, 4], Ymat[j, 4], Zmat[j, 4]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[i, 4 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the back wall pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1], pi[2] + q2[k%qn][k//qn]])
          # Quadrature point in the left wall pixel
          qpj = np.array([pj[0], pj[1] + q1[l%qn][l//qn], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[0] / r)
          cos_j = abs(difvec[1] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[i, 4 * n*n + j] = viewfactor  

print("Geometric view factors left->back done (4/15)")

# From the front wall (j) to the back wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the back wall
    pi = np.array([Xmat[i, 0], Ymat[i, 0], Zmat[i, 0]])
    # Centerpoint of the current pixel in the front wall
    pj = np.array([Xmat[j, 5], Ymat[j, 5], Zmat[j, 5]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[i, 5 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the back wall pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1], pi[2] + q2[k%qn][k//qn]])
          # Quadrature point in the front wall pixel
          qpj = np.array([pj[0] + q1[l%qn][l//qn], pj[1], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[1] / r)
          cos_j = abs(difvec[1] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[i, 5 * n*n + j] = viewfactor  

print("Geometric view factors front->back done (5/15)")

####################################
# From the floor (j) to the roof (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the roof
    pi = np.array([Xmat[i, 1], Ymat[i, 1], Zmat[i, 1]])
    # Centerpoint of the current pixel in the floor
    pj = np.array([Xmat[j, 2], Ymat[j, 2], Zmat[j, 2]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[n*n + i, 2 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the roof pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1] + q2[k%qn][k//qn], pi[2]])
          # Quadrature point in the floor pixel
          qpj = np.array([pj[0] + q1[l%qn][l//qn], pj[1] + q2[l%qn][l//qn], pj[2]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[2] / r)
          cos_j = abs(difvec[2] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[n*n + i, 2 * n*n + j] = viewfactor  

print("Geometric view factors floor->roof done (6/15)")

# From the right-hand-side wall (j) to the roof (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the roof
    pi = np.array([Xmat[i, 1], Ymat[i, 1], Zmat[i, 1]])
    # Centerpoint of the current pixel in the right wall
    pj = np.array([Xmat[j, 3], Ymat[j, 3], Zmat[j, 3]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[n*n + i, 3 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the roof pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1] + q2[k%qn][k//qn], pi[2]])
          # Quadrature point in the right wall pixel
          qpj = np.array([pj[0], pj[1] + q1[l%qn][l//qn], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[0] / r)
          cos_j = abs(difvec[2] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[n*n + i, 3 * n*n + j] = viewfactor  

print("Geometric view factors right->roof done (7/15)")

# From the left-hand-side wall (j) to the roof (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the roof
    pi = np.array([Xmat[i, 1], Ymat[i, 1], Zmat[i, 1]])
    # Centerpoint of the current pixel in the left wall
    pj = np.array([Xmat[j, 4], Ymat[j, 4], Zmat[j, 4]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[n*n + i, 4 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the roof pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1] + q2[k%qn][k//qn], pi[2]])
          # Quadrature point in the left wall pixel
          qpj = np.array([pj[0], pj[1] + q1[l%qn][l//qn], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[0] / r)
          cos_j = abs(difvec[2] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[n*n + i, 4 * n*n + j] = viewfactor  

print("Geometric view factors left->roof done (8/15)")

# From the front wall (j) to the roof (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the roof
    pi = np.array([Xmat[i, 1], Ymat[i, 1], Zmat[i, 1]])
    # Centerpoint of the current pixel in the front wall
    pj = np.array([Xmat[j, 5], Ymat[j, 5], Zmat[j, 5]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[n*n + i, 5 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the roof pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1] + q2[k%qn][k//qn], pi[2]])
          # Quadrature point in the front wall pixel
          qpj = np.array([pj[0] + q1[l%qn][l//qn], pj[1], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[2] / r)
          cos_j = abs(difvec[1] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[n*n + i, 5 * n*n + j] = viewfactor  

print("Geometric view factors front->roof done (9/15)")

# From the right-hand-side wall (j) to the floor (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the floor
    pi = np.array([Xmat[i, 2], Ymat[i, 2], Zmat[i, 2]])
    # Centerpoint of the current pixel in the right wall
    pj = np.array([Xmat[j, 3], Ymat[j, 3], Zmat[j, 3]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[2 * n*n + i, 3 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the floor pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1] + q2[k%qn][k//qn], pi[2]])
          # Quadrature point in the right wall pixel
          qpj = np.array([pj[0], pj[1] + q1[l%qn][l//qn], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[0] / r)
          cos_j = abs(difvec[2] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[2 * n*n + i, 3 * n*n + j] = viewfactor  

print("Geometric view factors right->floor done (10/15)")

# From the left-hand-side wall (j) to the floor (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the floor
    pi = np.array([Xmat[i, 2], Ymat[i, 2], Zmat[i, 2]])
    # Centerpoint of the current pixel in the left wall
    pj = np.array([Xmat[j, 4], Ymat[j, 4], Zmat[j, 4]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[2 * n*n + i, 4 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the floor pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1] + q2[k%qn][k//qn], pi[2]])
          # Quadrature point in the left wall pixel
          qpj = np.array([pj[0], pj[1] + q1[l%qn][l//qn], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[2] / r)
          cos_j = abs(difvec[0] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[2 * n*n + i, 4 * n*n + j] = viewfactor  

print("Geometric view factors left->floor done (11/15)")

# From the front wall (j) to the floor (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the floor
    pi = np.array([Xmat[i, 2], Ymat[i, 2], Zmat[i, 2]])
    # Centerpoint of the current pixel in the front wall
    pj = np.array([Xmat[j, 5], Ymat[j, 5], Zmat[j, 5]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[2 * n*n + i, 5 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the floor pixel
          qpi = np.array([pi[0] + q1[k%qn][k//qn], pi[1] + q2[k%qn][k//qn], pi[2]])
          # Quadrature point in the front wall pixel
          qpj = np.array([pj[0] + q1[l%qn][l//qn], pj[1], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[2] / r)
          cos_j = abs(difvec[1] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[2 * n*n + i, 5 * n*n + j] = viewfactor  

print("Geometric view factors front->floor done (12/15)")

##################################################################
# From the left-hand-side wall (j) to the right-hand-side-wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the right wall
    pi = np.array([Xmat[i, 3], Ymat[i, 3], Zmat[i, 3]])
    # Centerpoint of the current pixel in the left wall
    pj = np.array([Xmat[j, 4], Ymat[j, 4], Zmat[j, 4]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[3 * n*n + i, 4 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the right wall pixel
          qpi = np.array([pi[0], pi[1] + q1[k%qn][k//qn], pi[2] + q2[k%qn][k//qn]])
          # Quadrature point in the left wall pixel
          qpj = np.array([pj[0], pj[1] + q1[l%qn][l//qn], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[0] / r)
          cos_j = abs(difvec[0] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[3 * n*n + i, 4 * n*n + j] = viewfactor  

print("Geometric view factors left->right done (13/15)")

# From the front wall (j) to the right-hand-side-wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the right wall
    pi = np.array([Xmat[i, 3], Ymat[i, 3], Zmat[i, 3]])
    # Centerpoint of the current pixel in the front wall
    pj = np.array([Xmat[j, 5], Ymat[j, 5], Zmat[j, 5]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[3 * n*n + i, 5 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the right wall pixel
          qpi = np.array([pi[0], pi[1] + q1[k%qn][k//qn], pi[2] + q2[k%qn][k//qn]])
          # Quadrature point in the front wall pixel
          qpj = np.array([pj[0] + q1[l%qn][l//qn], pj[1], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[0] / r)
          cos_j = abs(difvec[1] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[3 * n*n + i, 5 * n*n + j] = viewfactor  

print("Geometric view factors front->right done (14/15)")

# From the front wall (j) to the left-hand-side-wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the left wall
    pi = np.array([Xmat[i, 4], Ymat[i, 4], Zmat[i, 4]])
    # Centerpoint of the current pixel in the front wall
    pj = np.array([Xmat[j, 5], Ymat[j, 5], Zmat[j, 5]])
    # Distance between the points
    difvec0 = pi - pj
    x = difvec0[0]
    y = difvec0[1]
    z = difvec0[2]
    r0 = math.sqrt(x*x + y*y + z*z)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[4 * n*n + i, 5 * n*n + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn*qn, qn*qn))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn*qn):
        for l in range(0, qn*qn):
          # Quadrature point in the left wall pixel
          qpi = np.array([pi[0], pi[1] + q1[k%qn][k//qn], pi[2] + q2[k%qn][k//qn]])
          # Quadrature point in the front wall pixel
          qpj = np.array([pj[0] + q1[l%qn][l//qn], pj[1], pj[2] + q2[l%qn][l//qn]])
          # Vector connecting the quadrature points
          difvec = qpi - qpj
          x = difvec[0]
          y = difvec[1]
          z = difvec[2]
          r = math.sqrt(x*x + y*y + z*z)
          cos_i = abs(difvec[0] / r)
          cos_j = abs(difvec[1] / r)
          # Evaluate integrand
          intgndmat[k, l] = (cos_i * cos_j) / (3.1415926 * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[4 * n*n + i, 5 * n*n + j] = viewfactor  

print("Geometric view factors front->left done (15/15)")

end = time.time()
print("View factors calculated in", end-start, "seconds")
# Use symmetry to finish the construction of F.
# F is symmetric since all the pixels in our model have equal size.

F = F + np.transpose(F)

# Check the matrix f, the row sums should all be one
print("Check values: all should ideally be one")
print(sum(F))

# Save matrix to disc
print("Saving to file...")
with open("../data/F_emptyroom", 'wb') as f:
  np.save(f, F)
  np.save(f, n)
  np.save(f, qn)
  np.save(f, d)
  np.save(f, Xmat)
  np.save(f, Ymat)
  np.save(f, Zmat)
print("Data saved!")
