
using LinearAlgebra, KernelMatrices, KernelMatrices.HODLR, NearestNeighbors, StaticArrays
import KernelMatrices: mtn_kernfun

const kfn = mtn_kernfun
const N   = 1024
const prm = [1.0, 5.0, 1.0]

# Generate points:
pts_sorted = KDTree([SVector{2, Float64}(rand(2).*10.0) for _ in 1:N]).data

# Take a subset of those points:
pts_subset = vcat(pts_sorted[1:100], pts_sorted[150:end])
pts_pred   = pts_sorted[101:149]

# Simulate data for all those points:
opts = maxlikopts(kernfun=kfn, prec=0.0, level=LogLevel(7), rank=72, 
                  saavecs=Vector{Vector}())
simd = HODLR.gpsimulate(pts_sorted, prm, opts, exact=false, kdtreesort=false)[2]

# Extract the data from the subset:
dat_subset = vcat(simd[1:100], simd[150:end])
dat_pred   = simd[101:149]

# Estimate via kriging the removed values:
kriged, kriged_var = HODLR.nys_krige(pts_pred, pts_subset, dat_subset, prm, opts)

