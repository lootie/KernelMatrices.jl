
using LinearAlgebra, StaticArrays, NLopt

# Declare the kernel function and its derivatives in the necessary forms for every worker:
using  KernelMatrices, KernelMatrices.HODLR
import KernelMatrices: mt1_kernfun, mt1_kernfun_d1, mt1_kernfun_d2
import KernelMatrices: mt1_kernfun_d1_d2, mt1_kernfun_d2_d2
kernfun  = mt1_kernfun
dfuns    = [mt1_kernfun_d1, mt1_kernfun_d2]

# Set the size of the simulated problem and generate the maximum likelihood options:
# (see the test_fit_trustregion.jl) for detailed annotations).
nsz     = 512
opts    = maxlikopts(kernfun=kernfun, dfuns=dfuns,
                     level=LogLevel(8), rank=72,
                     saavecs=HODLR.givesaa(35, nsz, seed=1618), verbose=true)

# simulate data points (or read in your not fake data points):
trueprms = [1.5, 5.0]
data     = HODLR.gpsimulate(kernfun, trueprms, nsz, 2, 100.0, exact=true)

# Estimate the kernel parameters using the above specified options:
initprms = [1.0, 3.0]
opt = Opt(:LD_MMA, 2)
min_objective!(opt, (x,p) -> HODLR.nll_objective(x, p, data, opts))
ftol_rel!(opt, 1.0e-5)
println("Optimizing:")
@time (minf, mle, ret) = optimize(opt, initprms)

# Compute the stochastic Fisher matrix at the estimated MLE:
println()
println("Stochastic fisher estimate at MLE took this much time:")
@time fish = HODLR.fisher_matrix(mle, data, opts)
mle_er     = sqrt.(diag(inv(fish))).*1.96
println()

# Print the output:
println("Results:")
println()
println("\t Truth:    Estimated (95% ±):")
println()
for j in eachindex(trueprms)
  println("\t $(round(trueprms[j], digits=4))       $(round(mle[j], digits=4)) ($(round(mle_er[j], digits=4)))")
end
println()
