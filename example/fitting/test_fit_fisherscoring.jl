
using Distributed, LinearAlgebra, StaticArrays, NLopt

# Declare the kernel function and its derivatives in the necessary forms for every worker:
@everywhere begin
  using  KernelMatrices, KernelMatrices.HODLR
  import KernelMatrices: mt1_kernfun, mt1_kernfun_d1, mt1_kernfun_d2
  import KernelMatrices: mt1_kernfun_d1_d2,mt1_kernfun_d2_d2
  kernfun  = mt1_kernfun
  dfuns    = [mt1_kernfun_d1, mt1_kernfun_d2]
  d2funs   = [[HODLR.ZeroFunction(), mt1_kernfun_d1_d2], [mt1_kernfun_d2_d2]]
end

# Set the size of the simulated problem and generate the maximum likelihood options:
nsz     = 512
opts    = maxlikopts(kernfun=kernfun, dfuns=dfuns, level=LogLevel(8),
                     rank=72, saavecs=HODLR.givesaa(35, nsz, seed=1618), verbose=true)

# simulate data points (or read in your not fake data points):
truprms  = [1.5, 5.0]
data     = HODLR.gpsimulate(kernfun, truprms, nsz, 2, 100.0, exact=true)

# Estimate the kernel parameters using the above specified options:
println("Optimizing:")
@time mle, fish = HODLR.fisherscore([1.0, 3.0], data, opts,
                             vrb=true, g_tol=1.0e-3, s_tol=1.0e-5)
mle_er   = sqrt.(diag(inv(fish))).*1.96
println()

# Print the output:
println("Results:")
println()
println("\t Truth:    Estimated (95% ±):")
println()
for j in eachindex(truprms)
  println("\t $(round(truprms[j], digits=4))       $(round(mle[j], digits=4)) ($(round(mle_er[j], digits=4)))")
end
println()
