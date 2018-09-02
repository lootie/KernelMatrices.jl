
using Distributed, Random

@everywhere begin
using LinearAlgebra, KernelMatrices, KernelMatrices.HODLR, StaticArrays, NearestNeighbors, NLopt, SpecialFunctions

# Load in the scripts and data files:
include("../fitting/fitting_funs.jl")
include("../fitting/generic_exact_functions.jl")

# Declare the kernel function and its derivatives in the necessary forms:
kernfun  = mt1_kernfun
dfuns    = [mt1_kernfun_d1, mt1_kernfun_d2]
d2funs   = [[mt1_kernfun_d1_d1, mt1_kernfun_d1_d2], [mt1_kernfun_d2_d2]]
end

# Set the seed for the same output each time:
Random.seed!(12345)

# Set some HODLR options:
opts = HODLR.Maxlikopts(
  kernfun,           # Kernel function
  dfuns,             # derivative functions
  0.0,               # The pointwise precision for the off-diagonal blocks. Not used for Nystrom method.
  HODLR.LogLevel(8), # The number of dyadic splits of the matrix dimensions. 0 leads to default value.
  64 ,               # The fixed rank of the off-diagonal blocks, with 0 meaning no maximum allowed rank.
  35 ,               # The number of symmetric bernoulli vectors used for stochastic estimates.
  false,             # Parallel flag for assembly, which is safe and very beneficial
  false,             # Parallel flag for factorization, which is less safe and beneficial.
  false,             # Verbose flag to see optimization path and fine-grained times
  0                  # Seed for the random sample vecs. 
)

# Choose the size of the data and its true parameters:
nsz      = 2^12
trup     = [3.0, 5.0]

# Simulate some data:
domsz    = 100.0
pts      = map(x->SVector{2, Float64}(rand(2).*domsz), 1:nsz)
simdd    = HODLR.gpsimulate(pts, trup, opts, exact=true, kdtreesort=true)
loc_s    = simdd[1]
dat_s    = simdd[2]

# Choose the levels to plot:
lvls     = collect(2:6)
gdsz     = 48
sprd     = 0.3
gd1      = linspace(trup[1]-sprd, trup[1]+sprd, gdsz)
gd2      = linspace(trup[2]-sprd, trup[2]+sprd, gdsz)
srfs     = map(x->zeros(gdsz, gdsz), 1:(length(lvls)+1))

# Get the exact likelihood surface:
@time srfs[1] .= HODLR.mapf(x->exact_nll_objective([x[1], x[2]], Float64[], loc_s, dat_s, kernfun, dfuns, false),
                            Iterators.product(gd1, gd2), nworkers(), true)

# Now loop over all the HODLR levels:
for j in eachindex(lvls)
  println("Computing for level $(lvls[j]) of $(lvls[end])...")
  topts = Hopts(kernfun,dfuns,0.0,lvls[j],72,35,false,false,ones(2).*0.1,false,1006)
  try
    @time srfs[j+1] .= HODLR.mapf(x->HODLR.nll_objective([x[1], x[2]], Float64[], loc_s, dat_s, topts),
                                  Iterators.product(gd1, gd2), nworkers(), true)
  catch
    println("Something failed...")
  end
end

# Now save everything:
using JLD
save("likelihood_surfaces.jld", "srfs", srfs, "data", dat_s, "locations", loc_s, "lvls", lvls,
     "gdsz", gdsz, "sprd", sprd, "trup", trup, "domsz", domsz)

