
using Distributed, Random

# Bring some things into scope on every worker:
@everywhere begin
using LinearAlgebra, KernelMatrices, KernelMatrices.HODLR, StaticArrays, SpecialFunctions

# Load in the scripts that define the generic functions:
include("../fitting/fitting_funs.jl")
include("../fitting/generic_exact_functions.jl")

# Choose the kernel function:
kernfun   = mt1_kernfun
dfuns     = [mt1_kernfun_d1, mt1_kernfun_d2]
d2funs    = [[HODLR.ZeroFunction(), mt1_kernfun_d1_d2], [mt1_kernfun_d2_d2]]
end

# Set the seed so that the numbers in the paper can be recreated exactly:
Random.seed!(31415)

# Some parameters for this particular simulation:
ranks     = [32, 72, 100]
nrept     = 3
powers    = 10:18
testp     = [2.0, 2.0]
trup      = [3.0, 5.0]

# Declare the matrix of norm differences:
hnl_time  = zeros(Float64, length(ranks), nrept, length(powers)) 
sgd_time  = zeros(Float64, length(ranks), nrept, length(powers)) 
shs_time  = zeros(Float64, length(ranks), nrept, length(powers)) 
xnl_time  = zeros(Float64, 4) 
xgd_time  = zeros(Float64, 4) 
xhs_time  = zeros(Float64, 4) 

# Loop across the three ranks to be studied:
for r in eachindex(ranks)
  # Update the fixed off-diagonal rank:
  println("Working on rank $(ranks[r]) now...")
  println()
  # Loop across, estimating/computing exactly for powers 2^j:
  for (j, jpow) in enumerate(powers)
    println("Working on size 2^$jpow...(rank $(ranks[r]))...($(nworkers()) workers)")
    # Generate opts that have the appropriate rank and SAA size:
    opts  = HODLR.Maxlikopts(kernfun,dfuns,0.0,HODLR.LogLevel(8),ranks[r],
                             HODLR.givesaa(35, 2^jpow), true, true, false, true)
    # Simulate that much data, without exact simulating because it doesn't matter here:
    pts   = map(x->StaticArrays.SVector{2, Float64}(rand(2).*100.0), 1:(2^jpow))
    simdd = HODLR.gpsimulate(pts, trup, opts, exact=false, kdtreesort=true)
    loc_s = simdd[1]
    dat_s = simdd[2]
    # Get the exact values for small sizes once:
    if jpow <= 13 && r == 1
      println()
      println("Computing exact values once for timing...")
      println()
      # Get the exact likelihood:
      xnl_time[j] = @elapsed exact_nll_objective(testp, Array{Float64}(undef, 0), loc_s, dat_s, kernfun, dfuns, false)
      # Get the exact gradient:
      xgd_time[j] = @elapsed exact_gradient(testp, loc_s, dat_s, kernfun, dfuns)
      # Get the exact hessian:
      xhs_time[j] = @elapsed exact_hessian(testp, loc_s, dat_s, kernfun, dfuns, d2funs)
    end
    # Now get samples for each:
    println("Computing stochastic values $nrept times for timing...")
    for k in 1:nrept
      println("Doing rep $k of $nrept...")
      # Get the HODLR likelihood:
      hnl_time[r,k,j] = @elapsed HODLR.nll_objective(testp, Array{Float64}(undef, 0), loc_s, dat_s, opts)
      # Get the stochastic gradients:
      sgd_time[r,k,j] = @elapsed HODLR.nll_gradient(testp, loc_s, dat_s, opts)
      # Get the stochastic Hessians:
      shs_time[r,k,j] = @elapsed HODLR.nll_hessian(testp, loc_s, dat_s, opts, d2funs)
    end
    println()
  end
end

# Save the scaling stuff:
using JLD
name = "../../data/scaling_times_"*string(nworkers())*"workers.jld"
save(name, "exact_lik_times", xnl_time,
           "exact_grd_times", xgd_time,
           "exact_hes_times", xhs_time,
           "hodlr_lik_times", hnl_time,
           "hodlr_grd_times", sgd_time,
           "hodlr_hes_times", shs_time)
println()
println("Saved file $name...")
println()

