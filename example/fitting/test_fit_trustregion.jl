
using Distributed, LinearAlgebra, StaticArrays

# Declare the kernel function and its derivatives in the necessary forms for every worker:
@everywhere begin
  using  KernelMatrices, KernelMatrices.HODLR
  import KernelMatrices: ps1_kernfun, ps1_kernfun_d2, ps1_kernfun_d2_d2 
  import KernelMatrices: sm1_kernfun, sm1_kernfun_d1, sm1_kernfun_d2
  import KernelMatrices: sm1_kernfun_d1_d2, sm1_kernfun_d2_d2
  kernfun  = ps1_kernfun 
  dfuns    = [ps1_kernfun_d2]
  d2funs   = Vector{Vector{Function}}(undef, 1) ; d2funs[1] = [ps1_kernfun_d2_d2]
  fkernfun = sm1_kernfun
  fdfuns   = [sm1_kernfun_d1, sm1_kernfun_d2]
  fd2funs  = [[HODLR.ZeroFunction(), sm1_kernfun_d1_d2], [sm1_kernfun_d2_d2]]
end

# Set the size of the simulated problem and generate the maximum likelihood options 
# for the PROFILE likelihood:
nsz     = 512
popts   = maxlikopts(
  kernfun = kernfun,     # Kernel function
  dfuns   = dfuns,       # derivative functions
  level   = LogLevel(8), # The level of the HODLR, set to log2(N) - 8.
  rank    = 72,          # The fixed rank of the off-diagonal blocks when using Nystrom.
  saavecs = HODLR.givesaa(35, nsz, seed=1618), # vectors for stochastic trace estimation.
  par_assem  = true,  # Parallel flag for assembly, which is always helpful.
  par_factor = true,  # Parallel flag for factorization, which is less obviously helpful.
  verbose    = true,  # Verbose flag to see optimization path and fine-grained times
  fix_saa    = true,  # flag for fixing SAA vectors.
)

# do the same thing for the full likelihood:
fopts         = deepcopy(popts) ; 
fopts.kernfun = fkernfun
fopts.dfuns   = fdfuns
fd2funs       = fd2funs

# simulate data points (or read in your not fake data points):
pts      = map(x->SVector{2, Float64}(rand(2).*100.0), 1:nsz)
trup     = [1.5, 5.0]
simdd    = HODLR.gpsimulate(pts, trup, fopts, exact=true, kdtreesort=true)
loc_s    = simdd[1]
dat_s    = simdd[2]

# Estimate the kernel parameters using the profile likelihood
println()
println("Optimizing profile likelihood...")
println()
@time prof_cnt, fitd = HODLR.trustregion(trup[2:end].-1.0, loc_s, dat_s, 
                                         d2funs, popts, profile=true, vrb=true)

# Now fit the full likelihood for comparison:
println()
println("Optimizing full likelihood...")
println()
@time fuld = HODLR.trustregion(trup.-1.0, loc_s, dat_s, fd2funs, fopts, vrb=true)[2]

# Print the output:
println("Results:")
println()
println("\t Profile:    Full:")
println()
for j in eachindex(trup)
  println("\t $(round(fitd[j], digits=4))       $(round(fuld[j], digits=4))")
end
println()

