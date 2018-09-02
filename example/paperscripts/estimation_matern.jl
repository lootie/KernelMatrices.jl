
# Load in the relevant packages, some on all workers:
using Distributed, Random, NLopt, JLD, NearestNeighbors, StatsBase
@everywhere using LinearAlgebra, KernelMatrices, KernelMatrices.HODLR, StaticArrays, SpecialFunctions

# Set the seed for reproducibility:
Random.seed!(27182)

# Load in and declare the kernel functions and exact loglik-related functions on all workers:
@everywhere begin
  include("../fitting/generic_exact_functions.jl")
  include("../fitting/fitting_funs.jl")
  kernfun   = ps1_kernfun
  dfuns     = Vector{Function}(1) ; dfuns[1] = ps1_kernfun_d2
  d2f       = Vector{Function}(1) ; d2f[1]   = ps1_kernfun_d2_d2
  d2funs    = [d2f]
  fkernfun  = sm1_kernfun
  fdfuns    = [sm1_kernfun_d1, sm1_kernfun_d2]
end

# Set some HODLR options:
hutchn  = 35
popts   = HODLR.Maxlikopts(kernfun, dfuns, 0.0,HODLR.LogLevel(8),72,HODLR.givesaa(hutchn,5),true,true,false,true)
fopts   = HODLR.Maxlikopts(fkernfun,fdfuns,0.0,HODLR.LogLevel(8),72,HODLR.givesaa(hutchn,5),true,true,false,true)

# Declare the sizes to fit:
nrep    = 5
j_range = 11:17
bigsz   = 2^j_range[end]
totlen  = length(j_range)

# Loop over the two scenarios;
for fnm in ["bigrange", "smallrange"]
  
  # Load in the data file, get the true parameters and initial values for optimization:
  datafile  = load("/path/to/matern_simulation_" * fnm * ".jld")
  trup      = datafile["stein_true"]
  pinit     = [trup[2]*0.8] # 20% in magnitude away from the true value.
  println()

  # Allocate the arrays to store things:
  exact_fit = Matrix{Vector{Float64}}(totlen,nrep)
  exact_nll = zeros(Float64, totlen, nrep)
  exact_hes = Matrix{Matrix{Float64}}(totlen, nrep)
  exact_nlt = zeros(Float64, totlen, nrep)
  exact_hlt = zeros(Float64, totlen, nrep)
  hodlr_fit = Matrix{Vector{Float64}}(totlen, nrep)
  hodlr_hes = Matrix{Matrix{Float64}}(totlen, nrep)
  hodlr_nlt = zeros(Float64, totlen, nrep)
  hodlr_hlt = zeros(Float64, totlen, nrep)

  for k in 1:nrep

    # Load in one of the very large datasets:
    println("Working with dataset $k...")
    println()
    sidx    = StatsBase.sample(1:bigsz, bigsz, replace=false)
    loc_f   = datafile["unsorted_locations"][sidx]
    dat_f   = datafile["unsorted_data"][k][sidx]

    # Now fit both exactly and HODLR-ly for small data sizes:
    for (j, jpow) in enumerate(j_range)

      # Extract the subset of the data:
      println("Trial $k: Fitting data of size 2^$jpow...")
      println()
      spts  = loc_f[1:(2^jpow)]
      sdts  = dat_f[1:(2^jpow)]
      loc_s = NearestNeighbors.KDTree(spts).data
      dat_s = sdts[KernelMatrices.getsortperm(spts, loc_s)]

      # Get new SAA vectors that are size-appropriate:
      fopts.saav .= HODLR.givesaa(hutchn, length(loc_s))
      popts.saav .= deepcopy(fopts.saav)

      # Fit the HODLR version, profiled:
      println("Optimizing accelerated method:")
      # if the range parameter is small, use the trust region method to speed up fitting:
      if fnm == "smallrange"
        h_t1 = @elapsed hminx = HODLR.trustregion(pinit, loc_s, dat_s, d2funs, popts, profile=true)
      # If the range parameter is big, use the method of moving asymptotes, gradient only:
      elseif fnm == "bigrange"
        opt = Opt(:LD_MMA, length(pinit))
        ftol_rel!(opt, 1.0e-8)
        min_objective!(opt, (p,g) -> HODLR.nlpl_objective(p, g, loc_s, dat_s, popts))
        h_t1 = @elapsed (hminf, hminx, hret) = optimize(opt, pinit)
        unshift!(hminx, HODLR.nlpl_scale(hminx, loc_s, dat_s, popts))
      else
        error("File name not recognized.")
      end
      println("That took $h_t1 seconds.")
      @show hminx
      println()
      println("Computing accelerated expected Fisher information matrix:")
      h_t2 = @elapsed hhess = HODLR.fisher_matrix(hminx, loc_s, dat_s, fopts)
      @show hhess
      println()

      # Store the relevant stuff:
      hodlr_fit[j,k] = hminx
      hodlr_hes[j,k] = hhess
      hodlr_nlt[j,k] = h_t1
      hodlr_hlt[j,k] = h_t2

      # If the size is sufficiently small, fit the exact version, too:
      if jpow <= 13

        # Start where the HODLR-likelihood left off:
        println("Optimizing exact method at accelerated minimizer:")
        opt = Opt(:LD_MMA, length(pinit))
        ftol_rel!(opt, 1.0e-8)
        min_objective!(opt, (p,g) -> exact_nlpl_objective(p, g, loc_s, dat_s, kernfun, dfuns, false))
        x_t1 = @elapsed (xminf, xminx, xret) = optimize(opt, hminx[2:end])
        unshift!(xminx, exact_nlpl_scale(xminx, loc_s, dat_s, kernfun))
        println("That took $x_t1 seconds.")
        @show xminx
        println()
        println("Computing exact Fisher information matrix:")
        x_t2 = @elapsed xhess = exact_fisher_matrix(xminx, loc_s, dat_s, fkernfun, fdfuns)
        println("That took $x_t2 seconds.")
        @show xhess
        println()

        # Store the relevant stuff:
        exact_fit[j,k] = xminx
        exact_hes[j,k] = xhess
        exact_nlt[j,k] = x_t1
        exact_hlt[j,k] = x_t2

      end

    end

  end

  # Save the data:
  outname = "estimates_matern_"*fnm*".jld" 
  save(outname, "exact_fit", exact_fit,
                "exact_nll", exact_nll,
                "exact_hes", exact_hes,
                "exact_nlt", exact_nlt,
                "exact_hlt", exact_hlt,
                "hodlr_fit", hodlr_fit,
                "hodlr_hes", hodlr_hes,
                "hodlr_nlt", hodlr_nlt,
                "hodlr_hlt", hodlr_hlt,
                "true_parm", trup)

end
