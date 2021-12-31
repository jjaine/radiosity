import numpy as np

# for printing, set precision and suppress scientific notation
np.set_printoptions(precision=5, suppress=True)

# mosaic resolution
n = 5

# integration quadrature parameter
qn = 3

# construct centerpoints of the mosaic tiles (patches)
# d denotes the length of the side of a patch
d = 2/n
tmp =  [-1-d/2 + i*d for i in range(1,n+1)]

# centerpoint coordinate matrices
Xmat = np.zeros((n**2, 6))
Ymat = np.zeros((n**2, 6))
Zmat = np.zeros((n**2, 6))

# construct the centerpoints for all the tiles in all the six walls

# the back wall (0)
X, Z = np.meshgrid(tmp, tmp)
Xmat[:,0] = np.concatenate(np.transpose(X)).flat
Zmat[:,0] = np.concatenate(np.transpose(Z)).flat
Ymat[:,0] = np.ones(n**2)

# roof (1)
X, Y = np.meshgrid(tmp, tmp)
Xmat[:,1] = np.concatenate(np.transpose(X)).flat
Ymat[:,1] = np.concatenate(np.transpose(Y)).flat
Zmat[:,1] = np.ones(n**2)

# floor (2)
Xmat[:,2] = np.concatenate(np.transpose(X)).flat
Ymat[:,2] = np.concatenate(np.transpose(Y)).flat
Zmat[:,2] = -np.ones(n**2)

# right-hand-side wall (3)
Y, Z = np.meshgrid(tmp, tmp)
Ymat[:,3] = np.concatenate(np.transpose(Y)).flat
Zmat[:,3] = np.concatenate(np.transpose(Z)).flat
Xmat[:,3] = np.ones(n**2)

# left-hand-side wall (4)
Ymat[:,4] = np.concatenate(np.transpose(Y)).flat
Zmat[:,4] = np.concatenate(np.transpose(Z)).flat
Xmat[:,4] = -np.ones(n**2)

# right-hand-side wall (5)
X, Z = np.meshgrid(tmp, tmp)
Xmat[:,5] = np.concatenate(np.transpose(X)).flat
Zmat[:,5] = np.concatenate(np.transpose(Z)).flat
Ymat[:,5] = -np.ones(n**2)

# formula for view factor between square-shaped pixels sharing an edge.
# from Cohen & Wallace: Radiosity and realistic image synthesis
# (Academic Press Professional 1993), Figure 4.4
atan = np.arctan
shared_edge_F = (2 * atan(1) - np.sqrt(2) * atan(1/np.sqrt(2) )+ .25 * np.log(3/4)) / np.pi

# quadrature points and weights for integrating over a square 
# of size d x d centered at the origin
tt = [i/qn*d - .5*d/qn - d/2 for i in range (1,qn+1)] 
q1, q2 = np.meshgrid(tt, tt)
qw = (d/qn) ** 4 # Area of quadrature pixel, squared, serves as the weight

# form the geometrical view factor matrix F.
# see http://en.wikipedia.org/wiki/View_factor for details of computation.

# Initialize the matrix
F = np.zeros((6*n**2, 6*n**2))

epsilon = 1 ** -8

# from the roof (j) to the back wall (u)
for i in range(0, n*n):
  for j in range(0, n*n):
    # centerpoint of the current pixel in the back wall
    pi = [Xmat[i, 0], Ymat[i, 0], Zmat[i, 0]]
    # centerpoint of the current pixel in the roof
    pj = [Xmat[i, 1], Ymat[i, 1], Zmat[i, 1]]
    # distance between the points
    difvec0 = [i-j for i, j in zip(pi, pj)]
    r0 = np.linalg.norm(difvec0)

    # check if the two pixels share an edge
    if r0 < np.sqrt(2) * d/2 + epsilon: # edge shared
      # calculate element of F analytically
      F[i, n*n + j] = shared_edge_F
    else: # edge not shared: integrate for F using quadrature
      # initalize matrix of integrand values at quadrature points
      intgndmat = np.zeros((qn**2, qn**2))
      # double loop over four-dimensional quadrature
      for k in range(0, qn**2):
        for l in range(0, qn**2):
          # quadrature point in the back wall pixel
          qpi = pi[:]
          qpi[0] += q1[k%qn][k//qn]
          qpi[2] += q2[k%qn][k//qn]
          # quadrature point in the roof pixel
          qpj = pj[:]
          qpj[0] += q1[l%qn][l//qn]
          qpj[1] += q2[l%qn][l//qn]
          #print(qpj)
          # vector connecting the quadrature points
          difvec = [qpi[i]-qpj[i] for i in range(0, len(qpi))]
          r = np.linalg.norm(difvec)
          tmp2 = difvec / r # unit direction vector
          cos_i = abs(tmp2[1])
          cos_j = abs(tmp2[2])
          # evaluate integrand
          intgndmat[k, l] = np.dot(cos_i, cos_j) / (np.pi * r**2)

      # calculate element of F
      viewfactor = qw * sum(sum(intgndmat)) / d**2
      F[i, n**2+j] = viewfactor
