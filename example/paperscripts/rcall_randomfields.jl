
using RCall, StaticArrays, SpecialFunctions, Random, LinearAlgebra

# A little function that converts the parameters from our modified Matern to the standard ones.
function stein_to_schlather(scale::Float64, range::Float64, nu::Float64)::Tuple{Float64, Float64}
  range_out = range/sqrt(2.0)
  scale_out = scale*(abs2(range)/(4.0*nu))*gamma(nu)*(2.0^(nu-1.0))
  return scale_out, range_out
end

# A wrapper for getting simulations via RandomFields.
function R_randomfieldmatern(parms::Vector, boxsize::Float64, powsz::Int64; dim::Int64=2,
                             seed::Union{Nothing, Int32}=nothing)
  # Get the size:
  sz = dim == 2 ? Int64(sqrt(powsz)) : Int64(cbrt(powsz))
  # Load the parms:
  scale, range, smooth = parms[1], parms[2], parms[3]
  # Load the library:
  reval("library(RandomFields)")
  # Set the seed if one is supplied;
  if seed != nothing
    reval("set.seed($seed)")
  end
  covstring = "RMmatern(var=$scale, scale=$range, nu=$smooth, notinvnu=TRUE)"
  grdstr    = "x=seq(0, $boxsize, length=$sz), y=seq(0, $boxsize, length=$sz)"
  if dim == 3 
    grdstr *= ", z=seq(0, $boxsize, length=$sz)"
  end
  ful = string("RFsimulate(", covstring, ",", grdstr, ")[1]")
  smd = reval(ful)
  # Get the corresponding spatial grid:
  gd  = LinRange(0.0, boxsize, sz)
  pts = ifelse(dim==2, map(x->SVector{2, Float64}(x[1], x[2]), Iterators.product(gd, gd))[:],
               map(x->SVector{3, Float64}(x[1], x[2], x[3]), Iterators.product(gd, gd, gd))[:])
  # Return both gridpoints and the output data:
  return (pts, rcopy(Array{Float64}, R"$smd$variable1"))
end





##
#
# Script component: simulating datasets to be fitted in the paper with maximal
# maximal focus on reproducibility. 
#
# I hoped that this would yield the same datasets every time and on any machine, but running it
# ~five months later definitely gave me different simulations back. So to use the exact simulated
# datasets in the paper, better to go to the JCGS submission page and download the data, or just
# email me (cgeoga@anl.gov) for it. 
#
##

using JLD

Random.seed!(111234)
stein_p1 = [3.0, 5.0,  1.0]
stein_p2 = [3.0, 50.0, 1.0]
seeds_p1 = rand(Int32, 5)
seeds_p2 = rand(Int32, 5)
names    = ["matern_simulation_smallrange.jld", "matern_simulation_bigrange.jld"]

for (prms, seeds, nam) in zip([stein_p1, stein_p2], [seeds_p1, seeds_p2], names)
  println("Simulating data for parameters $prms...")
  # Simulate the five datasets:
  pstd = vcat(collect(stein_to_schlather(prms...)), prms[end])
  data = map(x->R_randomfieldmatern(pstd, 100.0, 2^18, dim=2, seed=x), seeds)
  # save the files together:
  save(nam,  "unsorted_data", map(x->x[2], data), "unsorted_locations", data[1][1], "stein_true",
       prms, "standard_true", pstd)
end

mv(names[1], "../../data/"*names[1])
mv(names[2], "../../data/"*names[2])

