
@everywhere begin
using KernelMatrices, KernelMatrices.HODLR, StaticArrays, NearestNeighbors, NLopt, SpecialFunctions

# Set the seed for the same output each time:
srand(1618)

# Load in the covariance functions:
include("fitting_funs.jl")

# Declare the kernel function and its derivatives in the necessary forms:
kernfun  = sm1_kernfun
dfuns    = [sm1_kernfun_d1, sm1_kernfun_d2]
d2funs   = [[HODLR.ZeroFunction(), sm1_kernfun_d1_d2], [sm1_kernfun_d2_d2]]
end

# Choose the size of the problem:
nsz      = 1024

# Set some HODLR options:
opts = HODLR.Maxlikopts(
  kernfun,                # Kernel function
  dfuns,                  # derivative functions
  0.0,                    # The pointwise precision for the off-diagonal blocks. Not used for Nystrom method.
  0  ,                    # The number of dyadic splits of the matrix dimensions. 0 leads to default value.
  72 ,                    # The fixed rank of the off-diagonal blocks, with 0 meaning no maximum allowed rank.
  HODLR.givesaa(35, nsz), # The SAA vectors.
  true,                   # Parallel flag for assembly, which is safe and very beneficial
  true,                   # Parallel flag for factorization, which is less safe and beneficial.
  false ,                 # Verbose flag to see fine-grained times.
  true                    # The flag for fixing the SAA (as opposed to shuffling them).
)

# Simulate some data:
trup     = [1.25, 5.0]
dsz, dim = 100.0, 3
pts      = map(x->MVector{dim, Float64}(rand(dim).*dsz), 1:nsz)
simdd    = HODLR.gpsimulate(pts, trup, opts, exact=true, kdtreesort=true)

# K-D tree sorting:
loc_s    = simdd[1]
dat_s    = simdd[2]

# Hilbert space-filling curve sorting:
#loc_s    = KernelMatrices.hilbertsort(simdd[1])
#dat_s    = simdd[2][KernelMatrices.getsortperm(simdd[1], loc_s)]

# Choose some initial conditions:
inits    = trup .- [0.25, 1.0]

# Estimate the kernel parameters using the above specified options:
println("Optimizing...")
@time fitd = HODLR.trustregion(inits, loc_s, dat_s, d2funs, opts, vrb = true, rtol=1.0e-5)

# Compute the stochastic Fisher matrix at the estimated MLE:
println("Computing stochastic fisher information estimate...")
@time fsh  = HODLR.fisher_matrix(fitd, loc_s, dat_s, opts)
mle_er     = sqrt.(diag(inv(fsh))).*1.96

# Print the output:
println("Results:")
println()
println("\t Truth:    Estimated (95% ±):")
println()
for j in eachindex(trup)
  println("\t $(round(trup[j], 4))       $(round(fitd[j], 4)) ($(round(mle_er[j], 4)))")
end
println()
# =#
