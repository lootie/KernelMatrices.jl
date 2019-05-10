
using Distributed, Random

@everywhere begin
using LinearAlgebra, KernelMatrices, KernelMatrices.HODLR, StaticArrays, NearestNeighbors, NLopt, SpecialFunctions
import KernelMatrices: mt1_kernfun, mt1_kernfun_d1, mt1_kernfun_d2, mt1_kernfun_d1_d2,mt1_kernfun_d2_d2

# Set the seed for the same output each time:
Random.seed!(1618)

# Declare the kernel function and its derivatives in the necessary forms:
kernfun  = mt1_kernfun
dfuns    = [mt1_kernfun_d1, mt1_kernfun_d2]
d2funs   = [[HODLR.ZeroFunction(), mt1_kernfun_d1_d2], [mt1_kernfun_d2_d2]]
end

# Set the size:
nsz      = 512

# Set some HODLR options:
opts = HODLR.Maxlikopts(
  kernfun,               # Kernel function
  dfuns,                 # derivative functions
  0.0,                   # The pointwise precision for the off-diagonal blocks. Not used for Nystrom method.
  HODLR.LogLevel(8),     # The number of dyadic splits of the matrix dimensions, set here to log2(N) - 8.
  72 ,                   # The fixed rank of the off-diagonal blocks, with 0 meaning no maximum allowed rank.
  HODLR.givesaa(35, nsz),# Get the SAA vectors.
  true,                  # Parallel flag for assembly, which is safe and very beneficial
  true,                  # Parallel flag for factorization.
  true,                  # Verbose flag to see optimization path and fine-grained times
  true,                  # flag for fixing SAA vectors.
)

# Choose the size of the data and its true parameters:
trup     = [1.5, 5.0]

# Choose some parameters for two-stage fitting:
ep1      = 1.0e-5
ep2      = 1.0e-9
rnk2     = 72
hutch2   = 35
lvl2     = max(3, Int64(floor(log2(nsz))) - 8)
seed2    = 1958

# Simulate some data:
domsz    = 100.0
pts      = map(x->SVector{2, Float64}(rand(2).*domsz), 1:nsz)
simdd    = HODLR.gpsimulate(pts, trup, opts, exact=true, kdtreesort=true)
loc_s    = simdd[1]
dat_s    = simdd[2]

# Choose some initial conditions:
inits    = [1.0, 3.0] 

# Estimate the kernel parameters using the above specified options:
println("Optimizing:")
opt      = Opt(:LD_MMA, length(inits))
min_objective!(opt, (x,p) -> HODLR.nll_objective(x, p, loc_s, dat_s, opts))
ftol_rel!(opt, 1.0e-5)
@time (minf, minx, ret) = optimize(opt, inits)

# Compute the stochastic Hessian at the estimated MLE:
println()
println("Stochastic hessian estimate at MLE took this much time:")
@time hess = HODLR.nll_hessian(minx, loc_s, dat_s, opts, d2funs)
mle_er     = sqrt.(diag(inv(hess))).*1.96
println()

# Print the output:
println("Results:")
println()
println("\t Truth:    Estimated (95% ±):")
println()
for j in eachindex(trup)
  println("\t $(round(trup[j], digits=4))       $(round(minx[j], digits=4)) ($(round(mle_er[j], digits=4)))")
end
println()
