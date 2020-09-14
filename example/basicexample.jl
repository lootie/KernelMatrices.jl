
using Distributed, KernelMatrices, StaticArrays, NearestNeighbors, LinearAlgebra

# Get the points, so that K[i,j] = kernfun(pts[i], pts[k], parms):
dim      = 2
scal     = 10.0
N        = 2^10
pts      = map(x->StaticArrays.MVector{dim, Float64}(rand(dim).*scal), 1:N)

# Decide how to sort them, if any way (this is more relevant for the HODLR use case):
psorted  = NearestNeighbors.KDTree(pts).data   # K-D tree sort
#psorted = pts                                 # no sort

# Choose a kernel function, which I define @everywhere in case you want to
# parallelize in HODLR.jl:
function kernfun(x1, x2, parms)
  out = abs2(parms[1])/(1.0 + abs2(parms[2]*(norm(x1-x2))))
  if x1 == x2       # fudge factor
    out += 1.0e-12  # to stay
  end               # positive definite
  return out
end

# Choose some parms, which we allocate on the stack for marginally faster access:
kernprms = SVector{2, Float64}(3.0, 2.0)

# Declare the Kernel matrix:
K        = KernelMatrices.KernelMatrix(psorted, psorted, kernprms, kernfun)

# This struct K can now be treated a lot like a regular array. Examples of
# permissible operations:
  # K[j,k]
  # K[j,:]
  # K[:,k]
  # K*randn(N)

# To get the full matrix, use full(K). You can also do K*vec, but it will be
# substantially slower than full(K)*vec for some technical reasons. I would not
# recommend using the KernelMatrix struct in that way.

