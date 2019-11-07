
using Distributed, LinearAlgebra, StaticArrays, NLopt

# Declare the kernel function and its derivatives in the necessary forms for every worker:
@everywhere begin
  using  KernelMatrices, KernelMatrices.HODLR
  import KernelMatrices: mt1_kernfun, mt1_kernfun_d1, mt1_kernfun_d2, mt1_kernfun_d1_d2,mt1_kernfun_d2_d2
  kernfun  = mt1_kernfun
  dfuns    = [mt1_kernfun_d1, mt1_kernfun_d2]
  d2funs   = [[HODLR.ZeroFunction(), mt1_kernfun_d1_d2], [mt1_kernfun_d2_d2]]
end

# Set the size of the simulated problem and generate the maximum likelihood options:
nsz     = 512
opts    = HODLR.Maxlikopts(
  kernfun,           # Kernel function
  dfuns,             # derivative functions
  0.0,               # The pointwise precision for the off-diagonal blocks. Not used for Nystrom method.
  HODLR.LogLevel(8), # The number of dyadic splits of the matrix dimensions, set here to log2(N) - 8.
  72,                # The fixed rank of the off-diagonal blocks, with 0 meaning no maximum allowed rank.
  HODLR.givesaa(35, nsz, seed=1618), # Get the SAA vectors.
  true,  # Parallel flag for assembly, which is safe and very beneficial
  true,  # Parallel flag for factorization.
  true,  # Verbose flag to see optimization path and fine-grained times
  true,  # flag for fixing SAA vectors.
)

# simulate data points (or read in your not fake data points):
pts      = map(x->SVector{2, Float64}(rand(2).*100.0), 1:nsz)
trup     = [1.5, 5.0]
simdd    = HODLR.gpsimulate(pts, trup, opts, exact=true, kdtreesort=true)
loc_s    = simdd[1]
dat_s    = simdd[2]

# Estimate the kernel parameters using the above specified options:
println("Optimizing:")
@time mle, fish = HODLR.fisherscore([1.0, 3.0], loc_s, dat_s, opts,
                             vrb=true, g_tol=1.0e-3, s_tol=1.0e-5)
mle_er   = sqrt.(diag(inv(fish))).*1.96
println()

# Print the output:
println("Results:")
println()
println("\t Truth:    Estimated (95% ±):")
println()
for j in eachindex(trup)
  println("\t $(round(trup[j], digits=4))       $(round(mle[j], digits=4)) ($(round(mle_er[j], digits=4)))")
end
println()
