
using LinearAlgebra, StaticArrays, Ipopt

# Load in the interface file:
include("../interface/ipoptinterface.jl")

# Declare the kernel function and its derivatives in the necessary forms for every worker:
using  KernelMatrices, KernelMatrices.HODLR
import KernelMatrices: mt1_kernfun, mt1_kernfun_d1, mt1_kernfun_d2
import KernelMatrices: mt1_kernfun_d1_d2, mt1_kernfun_d2_d2
kernfun  = mt1_kernfun
dfuns    = [mt1_kernfun_d1, mt1_kernfun_d2]
d2funs   = [[HODLR.ZeroFunction(), mt1_kernfun_d1_d2], [mt1_kernfun_d2_d2]]

# Set the size of the simulated problem and generate the maximum likelihood options:
# (see the test_fit_trustregion.jl) for detailed annotations).
nsz     = 512
opts    = maxlikopts(kernfun=kernfun, dfuns=dfuns, d2funs=d2funs,
                     level=LogLevel(8), rank=72,
                     saavecs=HODLR.givesaa(35, nsz, seed=1618), verbose=false)

# simulate data points (or read in your not fake data points):
trueprms = [1.5, 5.0]
data     = HODLR.gpsimulate(kernfun, trueprms, nsz, 2, 100.0, exact=true)

# Estimate the kernel parameters using the above specified options:
initprms = [1.0, 3.0]
hess_type = :HESSIAN
prob = HODLR.Maxlikproblem(
  initprms,
  data, opts,
  lb=[0.0, 0.0],
  hess_type=hess_type
  )

# Scale the objective by its starting value and ignore the unscaled criteria
# Then tol is more or less the relative error
addOption(prob, "obj_scaling_factor", 1/HODLR.nll_objective(initprms, [], data, opts))
addOption(prob, "dual_inf_tol", 1e20)
addOption(prob, "compl_inf_tol", 1e20)
addOption(prob, "tol", 1e-3)

prob.x = initprms
println("Optimizing:")
@time status = solveProblem(prob)
mle = prob.x

# Compute the stochastic Hessian at the estimated MLE:
println()
println("Stochastic hessian estimate at MLE took this much time:")
@time hess = HODLR.nll_hessian(mle, data, opts)
mle_er     = sqrt.(diag(inv(hess))).*1.96
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
