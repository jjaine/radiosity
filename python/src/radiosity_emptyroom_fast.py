import numpy as np
import time

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
Xmat = np.zeros((n**2, 6))
Ymat = np.zeros((n**2, 6))
Zmat = np.zeros((n**2, 6))

# Construct the centerpoints for all the tiles in all the six walls

# The back wall (0)
X, Z = np.meshgrid(tmp, tmp)
Xmat[:,0] = np.concatenate(np.transpose(X)).flat
Zmat[:,0] = np.concatenate(np.transpose(Z)).flat
Ymat[:,0] = np.ones(n**2)

# Roof (1)
X, Y = np.meshgrid(tmp, tmp)
Xmat[:,1] = np.concatenate(np.transpose(X)).flat
Ymat[:,1] = np.concatenate(np.transpose(Y)).flat
Zmat[:,1] = np.ones(n**2)

# Floor (2)
Xmat[:,2] = np.concatenate(np.transpose(X)).flat
Ymat[:,2] = np.concatenate(np.transpose(Y)).flat
Zmat[:,2] = -np.ones(n**2)

# Right-hand-side wall (3)
Y, Z = np.meshgrid(tmp, tmp)
Ymat[:,3] = np.concatenate(np.transpose(Y)).flat
Zmat[:,3] = np.concatenate(np.transpose(Z)).flat
Xmat[:,3] = np.ones(n**2)

# Left-hand-side wall (4)
Ymat[:,4] = np.concatenate(np.transpose(Y)).flat
Zmat[:,4] = np.concatenate(np.transpose(Z)).flat
Xmat[:,4] = -np.ones(n**2)

# Front wall (5)
X, Z = np.meshgrid(tmp, tmp)
Xmat[:,5] = np.concatenate(np.transpose(X)).flat
Zmat[:,5] = np.concatenate(np.transpose(Z)).flat
Ymat[:,5] = -np.ones(n**2)

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
F = np.zeros((6*n**2, 6*n**2))

epsilon = 10 ** -8

start = time.time()

# Roof & floor, z vakio, indeksit 0 ja 1
# => 1 & 2
# LHW & RHW, x vakio, indeksit 1 ja 2
# => 3 & 4
# Front & back, y vakio, indeksit 0 ja 2
# => 0 & 5

# From everywhere (j) to the back wall (i)
for e in range(1, 6):
    for i in range(0, n*n):
        for j in range(0, n*n):
            # Centerpoint of the current pixel in the back wall
            pi = [Xmat[i, 0], Ymat[i, 0], Zmat[i, 0]]
            # Centerpoint of the current pixel in the other
            pj = [Xmat[j, e], Ymat[j, e], Zmat[j, e]]
            # Distance between the points
            difvec0 = [i-j for i, j in zip(pi, pj)]
            r0 = np.linalg.norm(difvec0)

            # Check if the two pixels share an edge
            if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
                # Calculate element of F analytically
                F[i, e * n**2 + j] = shared_edge_F
            else: # Edge not shared: integrate for F using quadrature
                # Initalize matrix of integrand values at quadrature points
                intgndmat = np.zeros((qn**2, qn**2))
                # Double loop over four-dimensional quadrature
                for k in range(0, qn**2):
                    for l in range(0, qn**2):
                        # Quadrature point in the back wall pixel
                        qpi = pi[:]
                        qpi[0] += q1[k%qn][k//qn]
                        qpi[2] += q2[k%qn][k//qn]
                        # Quadrature point in the other pixel
                        qpj = pj[:]
                        i1 = 0
                        i2 = 1
                        if e > 2 and e < 5:
                            i1 = 1
                            i2 = 2
                        if e > 4:
                            i2 = 2
                        qpj[i1] += q1[l%qn][l//qn]
                        qpj[i2] += q2[l%qn][l//qn]
                        # Vector connecting the quadrature points
                        difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
                        r = np.linalg.norm(difvec)
                        tmp2 = difvec / r # Unit direction vector
                        cos_i = abs(tmp2[1]) # TODO: FIXME
                        cos_j = abs(tmp2[2]) # TODO: FIXME
                        # Evaluate integrand
                        intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)

                # Calculate element of F
                viewfactor = qw * sum(sum(intgndmat)) / d**2
                F[i, e * n**2 + j] = viewfactor

print("Geometric view factors roof->back done (1/15)")
print("Geometric view factors floor->back done (2/15)")
print("Geometric view factors right->back done (3/15)")
print("Geometric view factors left->back done (4/15)")
print("Geometric view factors front->back done (5/15)")

####################################
# From the floor (j) to the roof (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the roof
    pi = [Xmat[i, 1], Ymat[i, 1], Zmat[i, 1]]
    # Centerpoint of the current pixel in the floor
    pj = [Xmat[j, 2], Ymat[j, 2], Zmat[j, 2]]
    # Distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[n**2 + i, 2 * n**2 + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # Quadrature point in the roof pixel
          qpi = pi[:]
          qpi[0] += q1[k%qn][k//qn]
          qpi[1] += q2[k%qn][k//qn]
          # Quadrature point in the floor pixel
          qpj = pj[:]
          qpj[0] += q1[l%qn][l//qn]
          qpj[1] += q2[l%qn][l//qn]
          # Vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # Unit direction vector
          cos_i = abs(tmp2[2])
          cos_j = abs(tmp2[2])
          # Evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[n**2 + i, 2 * n**2 + j] = viewfactor  

print("Geometric view factors floor->roof done (6/15)")

# From the right-hand-side wall (j) to the roof (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the roof
    pi = [Xmat[i, 1], Ymat[i, 1], Zmat[i, 1]]
    # Centerpoint of the current pixel in the right wall
    pj = [Xmat[j, 3], Ymat[j, 3], Zmat[j, 3]]
    # Distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[n**2 + i, 3 * n**2 + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # Quadrature point in the roof pixel
          qpi = pi[:]
          qpi[0] += q1[k%qn][k//qn]
          qpi[1] += q2[k%qn][k//qn]
          # Quadrature point in the right wall pixel
          qpj = pj[:]
          qpj[1] += q1[l%qn][l//qn]
          qpj[2] += q2[l%qn][l//qn]
          # Vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # Unit direction vector
          cos_i = abs(tmp2[0])
          cos_j = abs(tmp2[2])
          # Evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[n**2 + i, 3 * n**2 + j] = viewfactor  

print("Geometric view factors right->roof done (7/15)")

# From the left-hand-side wall (j) to the roof (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the roof
    pi = [Xmat[i, 1], Ymat[i, 1], Zmat[i, 1]]
    # Centerpoint of the current pixel in the left wall
    pj = [Xmat[j, 4], Ymat[j, 4], Zmat[j, 4]]
    # Distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[n**2 + i, 4 * n**2 + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # Quadrature point in the roof pixel
          qpi = pi[:]
          qpi[0] += q1[k%qn][k//qn]
          qpi[1] += q2[k%qn][k//qn]
          # Quadrature point in the left wall pixel
          qpj = pj[:]
          qpj[1] += q1[l%qn][l//qn]
          qpj[2] += q2[l%qn][l//qn]
          # Vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # Unit direction vector
          cos_i = abs(tmp2[0])
          cos_j = abs(tmp2[2])
          # Evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[n**2 + i, 4 * n**2 + j] = viewfactor  

print("Geometric view factors left->roof done (8/15)")

# From the front wall (j) to the roof (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the roof
    pi = [Xmat[i, 1], Ymat[i, 1], Zmat[i, 1]]
    # Centerpoint of the current pixel in the front wall
    pj = [Xmat[j, 5], Ymat[j, 5], Zmat[j, 5]]
    # Distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[n**2 + i, 5 * n**2 + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # Quadrature point in the roof pixel
          qpi = pi[:]
          qpi[0] += q1[k%qn][k//qn]
          qpi[1] += q2[k%qn][k//qn]
          # Quadrature point in the front wall pixel
          qpj = pj[:]
          qpj[0] += q1[l%qn][l//qn]
          qpj[2] += q2[l%qn][l//qn]
          # Vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # Unit direction vector
          cos_i = abs(tmp2[2])
          cos_j = abs(tmp2[1])
          # Evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[n**2 + i, 5 * n**2 + j] = viewfactor  

print("Geometric view factors front->roof done (9/15)")

# From the right-hand-side wall (j) to the floor (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the floor
    pi = [Xmat[i, 2], Ymat[i, 2], Zmat[i, 2]]
    # Centerpoint of the current pixel in the right wall
    pj = [Xmat[j, 3], Ymat[j, 3], Zmat[j, 3]]
    # Distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[2 * n**2 + i, 3 * n**2 + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # Quadrature point in the floor pixel
          qpi = pi[:]
          qpi[0] += q1[k%qn][k//qn]
          qpi[1] += q2[k%qn][k//qn]
          # Quadrature point in the right wall pixel
          qpj = pj[:]
          qpj[1] += q1[l%qn][l//qn]
          qpj[2] += q2[l%qn][l//qn]
          # Vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # Unit direction vector
          cos_i = abs(tmp2[0])
          cos_j = abs(tmp2[2])
          # Evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[2 * n**2 + i, 3 * n**2 + j] = viewfactor  

print("Geometric view factors right->floor done (10/15)")

# From the left-hand-side wall (j) to the floor (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the floor
    pi = [Xmat[i, 2], Ymat[i, 2], Zmat[i, 2]]
    # Centerpoint of the current pixel in the left wall
    pj = [Xmat[j, 4], Ymat[j, 4], Zmat[j, 4]]
    # Distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[2 * n**2 + i, 4 * n**2 + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # Quadrature point in the floor pixel
          qpi = pi[:]
          qpi[0] += q1[k%qn][k//qn]
          qpi[1] += q2[k%qn][k//qn]
          # Quadrature point in the left wall pixel
          qpj = pj[:]
          qpj[1] += q1[l%qn][l//qn]
          qpj[2] += q2[l%qn][l//qn]
          # Vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # Unit direction vector
          cos_i = abs(tmp2[2])
          cos_j = abs(tmp2[0])
          # Evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[2 * n**2 + i, 4 * n**2 + j] = viewfactor  

print("Geometric view factors left->floor done (11/15)")

# From the front wall (j) to the floor (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the floor
    pi = [Xmat[i, 2], Ymat[i, 2], Zmat[i, 2]]
    # Centerpoint of the current pixel in the front wall
    pj = [Xmat[j, 5], Ymat[j, 5], Zmat[j, 5]]
    # Distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[2 * n**2 + i, 5 * n**2 + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # Quadrature point in the floor pixel
          qpi = pi[:]
          qpi[0] += q1[k%qn][k//qn]
          qpi[1] += q2[k%qn][k//qn]
          # Quadrature point in the front wall pixel
          qpj = pj[:]
          qpj[0] += q1[l%qn][l//qn]
          qpj[2] += q2[l%qn][l//qn]
          # Vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # Unit direction vector
          cos_i = abs(tmp2[2])
          cos_j = abs(tmp2[1])
          # Evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[2 * n**2 + i, 5 * n**2 + j] = viewfactor  

print("Geometric view factors front->floor done (12/15)")

##################################################################
# From the left-hand-side wall (j) to the right-hand-side-wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the right wall
    pi = [Xmat[i, 3], Ymat[i, 3], Zmat[i, 3]]
    # Centerpoint of the current pixel in the left wall
    pj = [Xmat[j, 4], Ymat[j, 4], Zmat[j, 4]]
    # Distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[3 * n**2 + i, 4 * n**2 + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # Quadrature point in the right wall pixel
          qpi = pi[:]
          qpi[1] += q1[k%qn][k//qn]
          qpi[2] += q2[k%qn][k//qn]
          # Quadrature point in the left wall pixel
          qpj = pj[:]
          qpj[1] += q1[l%qn][l//qn]
          qpj[2] += q2[l%qn][l//qn]
          # Vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # Unit direction vector
          cos_i = abs(tmp2[0])
          cos_j = abs(tmp2[0])
          # Evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[3 * n**2 + i, 4 * n**2 + j] = viewfactor  

print("Geometric view factors left->right done (13/15)")

# From the front wall (j) to the right-hand-side-wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the right wall
    pi = [Xmat[i, 3], Ymat[i, 3], Zmat[i, 3]]
    # Centerpoint of the current pixel in the front wall
    pj = [Xmat[j, 5], Ymat[j, 5], Zmat[j, 5]]
    # Distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[3 * n**2 + i, 5 * n**2 + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # Quadrature point in the right wall pixel
          qpi = pi[:]
          qpi[1] += q1[k%qn][k//qn]
          qpi[2] += q2[k%qn][k//qn]
          # Quadrature point in the front wall pixel
          qpj = pj[:]
          qpj[0] += q1[l%qn][l//qn]
          qpj[2] += q2[l%qn][l//qn]
          # Vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # Unit direction vector
          cos_i = abs(tmp2[0])
          cos_j = abs(tmp2[1])
          # Evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[3 * n**2 + i, 5 * n**2 + j] = viewfactor  

print("Geometric view factors front->right done (14/15)")

# From the front wall (j) to the left-hand-side-wall (i)
for i in range(0, n*n):
  for j in range(0, n*n):
    # Centerpoint of the current pixel in the left wall
    pi = [Xmat[i, 4], Ymat[i, 4], Zmat[i, 4]]
    # Centerpoint of the current pixel in the front wall
    pj = [Xmat[j, 5], Ymat[j, 5], Zmat[j, 5]]
    # Distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # Check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # Edge shared
      # Calculate element of F analytically
      F[4 * n**2 + i, 5 * n**2 + j] = shared_edge_F
    else: # Edge not shared: integrate for F using quadrature
      # Initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # Double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # Quadrature point in the left wall pixel
          qpi = pi[:]
          qpi[1] += q1[k%qn][k//qn]
          qpi[2] += q2[k%qn][k//qn]
          # Quadrature point in the front wall pixel
          qpj = pj[:]
          qpj[0] += q1[l%qn][l//qn]
          qpj[2] += q2[l%qn][l//qn]
          # Vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # Unit direction vector
          cos_i = abs(tmp2[0])
          cos_j = abs(tmp2[1])
          # Evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)
          
      # Calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[4 * n**2 + i, 5 * n**2 + j] = viewfactor  

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
