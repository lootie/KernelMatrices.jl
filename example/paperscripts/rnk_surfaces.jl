
using Distributed, Random, ProgressMeter

@everywhere begin
using LinearAlgebra, KernelMatrices, KernelMatrices.HODLR, StaticArrays, NearestNeighbors, NLopt, SpecialFunctions
import KernelMatrices: mt1_kernfun, mt1_kernfun_d1, mt1_kernfun_d2, mt1_kernfun_d1_d2, mt1_kernfun_d2_d2

# Declare the kernel function and its derivatives in the necessary forms:
kernfun  = mt1_kernfun
dfuns    = [mt1_kernfun_d1, mt1_kernfun_d2]
d2funs   = [[HODLR.ZeroFunction(), mt1_kernfun_d1_d2], [mt1_kernfun_d2_d2]]
end

# Set the seed for the same output each time:
Random.seed!(12345)

# dummy opts for the simulation:
dopts    = HODLR.Maxlikopts(kernfun,dfuns,0.0,HODLR.FixedLevel(0),2,HODLR.givesaa(35,5),false,false,false,true)

# Choose the size of the data and its true parameters:
nsz      = 2^12
trup     = [3.0, 5.0]
lvl      = 4

# Simulate some data:
domsz    = 100.0
pts      = map(x->SVector{2, Float64}(rand(2).*domsz), 1:nsz)
simdd    = HODLR.gpsimulate(pts, trup, dopts, exact=true, kdtreesort=true)
loc_s    = simdd[1]
dat_s    = simdd[2]

# Choose the levels to plot:
rnks     = [0, 2, 36, 48, 72, 96]
gdsz     = 48
sprd     = 0.35
gd1      = LinRange(trup[1]-sprd, trup[1]+sprd, gdsz)
gd2      = LinRange(trup[2]-sprd, trup[2]+sprd, gdsz)
srfs     = map(x->zeros(gdsz, gdsz), eachindex(rnks))
grd      = collect(Iterators.product(gd1, gd2))

# Now loop over all the HODLR levels:
for j in eachindex(rnks)
  println("Computing for $j-th rank of $(length(rnks))...")
  if j == 1
    topts = HODLR.Maxlikopts(kernfun,dfuns,0.0,HODLR.FixedLevel(0),2,
                             HODLR.givesaa(35,5),false,false,false,true)
  else
    topts = HODLR.Maxlikopts(kernfun,dfuns,0.0,HODLR.FixedLevel(lvl),rnks[j],
                             HODLR.givesaa(35,5),false,false,false,true)
  end
  try
  @time srfs[j] .= @showprogress pmap(x->HODLR.nll_objective([x[1], x[2]], Float64[], loc_s, dat_s, topts), grd)
  catch
    println("Something failed...")
  end
end

# Check the exact likelihood:
exact_5x5 = HODLR.mapf(x->HODLR.exact_nll_objective([x[1], x[2]], Float64[], loc_s, dat_s, kernfun, dfuns, false),
                            grd[1:5, 1:5], nworkers(), true)
@show isapprox(srfs[1][1:5, 1:5], exact_5x5, rtol=1.0e-14)

# Now save everything:
using JLD
save("../../data/likelihood_surfaces_rnk.jld", "srfs", srfs, "data", dat_s, "locations", loc_s,
     "rnks", rnks, "gdsz", gdsz, "sprd", sprd, "trup", trup, "domsz", domsz)
