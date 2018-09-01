
using Distributed, Random

@everywhere begin
using LinearAlgebra, KernelMatrices, KernelMatrices.HODLR, StaticArrays, NearestNeighbors, NLopt, SpecialFunctions

# Set the seed for the same output each time:
Random.seed!(1618)

# Load in the covariance functions:
include("fitting_funs.jl")

# Declare the kernel function and its derivatives in the necessary forms for the profile likelihood:
kernfun  = ps1_kernfun 
dfuns    = Vector{Function}(undef, 1) ; dfuns[1] = ps1_kernfun_d2
d2f      = Vector{Function}(undef, 1) ; d2f[1]   = ps1_kernfun_d2_d2
d2funs   = [d2f]
end

# Choose size, true parameters, and init stuff:
nsz      = 1024
trup     = [3.0, 5.0]
finits   = trup.-1.0
pinits   = trup[2:end].-1.0

# Set the HODLR options for the profile likelihood:
popts = HODLR.Maxlikopts(
  kernfun,                # Kernel function
  dfuns,                  # derivative functions
  0.0,                    # The pointwise precision for the off-diagonal blocks. Not used for Nystrom method.
  0  ,                    # The number of dyadic splits of the matrix dimensions. 0 leads to default value.
  72 ,                    # The fixed rank of the off-diagonal blocks, with 0 meaning no maximum allowed rank.
  HODLR.givesaa(35, nsz), # SAA vectors.
  true,                   # Parallel flag for assembly, which is safe and very beneficial
  true,                   # Parallel flag for factorization, which is less safe and beneficial.
  false,                  # Verbose flag to see optimization path and fine-grained times
  true,                   # Flag for fixing SAA vectors.
)

# Set equivalent options for the full likelihood:
fopts         = deepcopy(popts) ; 
fopts.kernfun = sm1_kernfun
fopts.dfuns   = [sm1_kernfun_d1, sm1_kernfun_d2]
fd2funs       = [[HODLR.ZeroFunction(), sm1_kernfun_d1_d2],
                 [sm1_kernfun_d2_d2],
                ]

# Simulate some data:
domsz    = 100.0
pts      = map(x->SVector{2, Float64}(rand(2).*domsz), 1:nsz)
simdd    = HODLR.gpsimulate(pts, trup, fopts, exact=true, kdtreesort=true)
loc_s    = simdd[1]
dat_s    = simdd[2]

# Estimate the kernel parameters using the profile likelihood
println()
println("Optimizing profile likelihood...")
println()
@time fitd = HODLR.trustregion(pinits, loc_s, dat_s, d2funs, popts, vrb=true, profile=true)

# Now fit the full likelihood for comparison:
println()
println("Optimizing full likelihood...")
println()
@time fuld = HODLR.trustregion(finits, loc_s, dat_s, fd2funs, fopts, vrb=true)

# Print the output:
println("Results:")
println()
println("\t Profile:    Full:")
println()
for j in eachindex(trup)
  println("\t $(round(fitd[j], digits=4))       $(round(fuld[j], digits=4))")
end
println()
