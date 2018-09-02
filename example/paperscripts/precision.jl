
using Distributed, Random

@everywhere begin
using LinearAlgebra, KernelMatrices, KernelMatrices.HODLR, StaticArrays, NearestNeighbors, JLD, SpecialFunctions

# Load in the scripts that define the generic functions:
include("../fitting/fitting_funs.jl")
include("../fitting/generic_exact_functions.jl")

# Choose the kernel function:
kernfun   = sm1_kernfun
dfuns     = [sm1_kernfun_d1, sm1_kernfun_d2]
d2funs    = [[HODLR.ZeroFunction(), sm1_kernfun_d1_d2], [sm1_kernfun_d2_d2]]
end

# Set the seed so that the numbers in the paper can be recreated exactly:
Random.seed!(31415)

# Some parameters for this particular simulation:
nrept     = 5
powers    = 10:13
testp     = [2.0, 2.0]
trup      = [3.0, 5.0]

# Declare the matrix of norm differences. Naming key: 
apx_fit_mle  = Matrix{Vector{Float64}}(nrept, length(powers))
ext_grd_mle  = Matrix{Vector{Float64}}(nrept, length(powers))
ext_grd_far  = Matrix{Vector{Float64}}(nrept, length(powers))
apx_grd_far  = Matrix{Vector{Float64}}(nrept, length(powers))
apx_grd_mle  = Matrix{Vector{Float64}}(nrept, length(powers))
ext_hes_far  = Matrix{Matrix{Float64}}(nrept, length(powers))
ext_hes_mle  = Matrix{Matrix{Float64}}(nrept, length(powers))
ext_fsh_mle  = Matrix{Matrix{Float64}}(nrept, length(powers))
apx_hes_far  = Matrix{Matrix{Float64}}(nrept, length(powers))
apx_hes_mle  = Matrix{Matrix{Float64}}(nrept, length(powers))
apx_fsh_mle  = Matrix{Matrix{Float64}}(nrept, length(powers))

# Get the options in place:
opts = HODLR.Maxlikopts(kernfun,dfuns,0.0,HODLR.LogLevel(8),72,HODLR.givesaa(35, 5), true, true, false, true)

# Loop across, estimating/computing exactly for powers 2^j:
for (j, jpow) in enumerate(powers)
  println("Working on size 2^$jpow...")
  for k in 1:nrept
    println("Doing rep $k of $nrept, shuffling SAA vectors and simulating new data...")
    opts.saav       .= HODLR.givesaa(35, 2^jpow)
    # Extract that much of the data:
    pts              = map(x->StaticArrays.SVector{2, Float64}(rand(2).*100.0), 1:(2^jpow))
    simdd            = HODLR.gpsimulate(pts, trup, opts, exact=true, kdtreesort=true)
    loc_s,dat_s      = simdd[1], simdd[2]
    # Fit the data:
    apx_fit_mle[k,j] = HODLR.trustregion(trup.-1.0, loc_s, dat_s, d2funs, opts)
    # Get the exact HODLR hessian and gradient at the init and MLE:
    ext_grd_far[k,j] = exact_HODLR_gradient(testp,          loc_s, dat_s, opts)
    ext_grd_mle[k,j] = exact_HODLR_gradient(apx_fit_mle[j], loc_s, dat_s, opts)
    ext_fsh_mle[k,j] = exact_HODLR_fisher(apx_fit_mle[j],   loc_s, dat_s, opts)
    ext_hes_far[k,j] = exact_HODLR_hessian(testp,           loc_s, dat_s, opts, d2funs)
    ext_hes_mle[k,j] = exact_HODLR_hessian(apx_fit_mle[j],  loc_s, dat_s, opts, d2funs)
    # Get the stochastic gradient and Hessian at the init and MLE:
    apx_grd_far[k,j] = HODLR.nll_gradient(testp,            loc_s, dat_s, opts)
    apx_grd_mle[k,j] = HODLR.nll_gradient(apx_fit_mle[j],   loc_s, dat_s, opts)
    apx_fsh_mle[k,j] = HODLR.fisher_matrix(apx_fit_mle[j],  loc_s, dat_s, opts)
    apx_hes_far[k,j] = HODLR.nll_hessian(testp,             loc_s, dat_s, opts, d2funs)
    apx_hes_mle[k,j] = HODLR.nll_hessian(apx_fit_mle[j],    loc_s, dat_s, opts, d2funs)
  end
  println()
end

# Save the precision stuff:
save("precision_estimators.jld", "apx_fit_mle",  apx_fit_mle,
                                 "ext_grd_mle",  ext_grd_mle,
                                 "ext_grd_far",  ext_grd_far,
                                 "apx_grd_far",  apx_grd_far,
                                 "apx_grd_mle",  apx_grd_mle,
                                 "ext_hes_far",  ext_hes_far,
                                 "ext_hes_mle",  ext_hes_mle,
                                 "ext_fsh_mle",  ext_fsh_mle,
                                 "apx_hes_far",  apx_hes_far,
                                 "apx_hes_mle",  apx_hes_mle,
                                 "apx_fsh_mle",  apx_fsh_mle)

