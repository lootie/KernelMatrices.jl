
using Distributed, Random

@everywhere begin
using LinearAlgebra, KernelMatrices, KernelMatrices.HODLR, StaticArrays, NearestNeighbors, NLopt, SpecialFunctions

# Load in the scripts and data files:
include("../fitting/fitting_funs.jl")
include("../fitting/generic_exact_functions.jl")

# Declare the kernel function and its derivatives in the necessary forms:
kernfun  = mt1_kernfun
dfuns    = [mt1_kernfun_d1, mt1_kernfun_d2]
d2funs   = [[HODLR.ZeroFunction(), mt1_kernfun_d1_d2], [mt1_kernfun_d2_d2]]
end

# Set the seed for the same output each time:
Random.seed!(12345)

# Set some HODLR options:
opts = HODLR.Maxlikopts(kernfun,dfuns,0.0,0,64,HODLR.givesaa(35, 2^12),false,false,false,true)

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
ranks    = [2, 36, 48, 72, 96]
gdsz     = 48
sprd     = 0.3
gd1      = linspace(trup[1]-sprd, trup[1]+sprd, gdsz)
gd2      = linspace(trup[2]-sprd, trup[2]+sprd, gdsz)
ggd      = Iterators.product(gd1, gd2)
srfs     = map(x->zeros(gdsz, gdsz), 1:(length(ranks)+1))

# Get the exact likelihood surface:
@time srfs[1] .= HODLR.mapf(x->exact_nll_objective([x[1], x[2]], Float64[], loc_s, dat_s, kernfun, dfuns, false),
                            ggd, nworkers(), true)

# Now loop over all the HODLR levels:
for j in eachindex(ranks)
  println("Computing for rank $(ranks[j])...")
  topts = HODLR.Maxlikopts(kernfun,dfuns,0.0,0,ranks[j],HODLR.givesaa(35, 2^12),false,false,false,true)
  try
    @time srfs[j+1] .= HODLR.mapf(x->HODLR.nll_objective([x[1], x[2]], Float64[], loc_s, dat_s, topts), 
                                  ggd, nworkers(), true)
  catch
    println("Something failed...")
  end
end

# Now save everything:
using JLD
save("likelihood_surfaces_rank.jld", "srfs", srfs, "data", dat_s, "locations", loc_s, "ranks", ranks,
     "gdsz", gdsz, "sprd", sprd, "trup", trup, "domsz", domsz)

