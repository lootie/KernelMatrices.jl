
using LinearAlgebra, Random, KernelMatrices, KernelMatrices.HODLR, StaticArrays, NearestNeighbors, JLD, SpecialFunctions
include("../fitting/fitting_funs.jl")

# Set the seed for reproducibility:
Random.seed!(57721) 

# Choose the kernel function to use:
kfun = mtn_kernfun
dfns = [mtn_kernfun_d1, mtn_kernfun_d2, mtn_kernfun_d3]

# Write a quick little function for the un-symmetrized trace:
function HODLR_asymmetric_trace(HK::HODLR.KernelHODLR, HKj::HODLR.DerivativeHODLR, vecs)
  out = 0.0
  uv  = zeros(length(vecs))
  for j in eachindex(vecs)
    out += dot(vecs[j], HK\(HKj*vecs[j]))
  end
  return out/length(vecs)
end

# Declare various options for the test:
tryp     = [3.0, 40.0, 1.2]
psz      = 2^12
opts     = HODLR.Maxlikopts(kfun,dfns,0.0,HODLR.LogLevel(8),72,HODLR.givesaa(1,psz),false,false,false,false)
hutchN   = [1, 2, 3, 4, 5, 10, 20, 30, 50, 100]
sampleN  = 50

# Simulate some points:
pts      = NearestNeighbors.KDTree(map(x->SVector{2, Float64}(rand(2).*100.0), 1:psz)).data

# Assemble all the pieces:
K        = KernelMatrices.KernelMatrix(pts, pts, tryp, kfun)
HK       = HODLR.KernelHODLR(K, opts.epK, opts.mrnk, opts.lvl, nystrom=true)
HKj1     = HODLR.DerivativeHODLR(K, dfns[1], HK)
HKj2     = HODLR.DerivativeHODLR(K, dfns[2], HK)
HKj3     = HODLR.DerivativeHODLR(K, dfns[3], HK)
HKf      = full(HK)
HODLR.symmetricfactorize!(HK)

# compute the exact HODLR traces:
tapx_tr1 = trace(HKf\full(HKj1))
tapx_tr2 = trace(HKf\full(HKj2))
tapx_tr3 = trace(HKf\full(HKj3))
# Now get the estimated traces:
tr1_sym  = zeros(Float64, sampleN, length(hutchN))
tr1_asm  = zeros(Float64, sampleN, length(hutchN))
tr2_sym  = zeros(Float64, sampleN, length(hutchN))
tr2_asm  = zeros(Float64, sampleN, length(hutchN))
tr3_sym  = zeros(Float64, sampleN, length(hutchN))
tr3_asm  = zeros(Float64, sampleN, length(hutchN))
println()
for j in eachindex(hutchN)
  hN = hutchN[j]
  println("N = $hN:")
  opts.saav = HODLR.givesaa(hN, psz)
  println()
  hNtime = @elapsed begin
  for k in 1:sampleN
    HODLR.saa_shuffle!(opts.saav)
    println("doing rep $k out of $sampleN...")
    # Asymmetric estimates:
    tr1_asm[k,j] = HODLR_asymmetric_trace(HK, HKj1, opts.saav)
    tr2_asm[k,j] = HODLR_asymmetric_trace(HK, HKj2, opts.saav)
    tr3_asm[k,j] = HODLR_asymmetric_trace(HK, HKj3, opts.saav)
    # Symmetric estimates:
    tr1_sym[k,j] = HODLR.HODLR_trace_term(HK, HKj1, opts.saav, false)
    tr2_sym[k,j] = HODLR.HODLR_trace_term(HK, HKj2, opts.saav, false)
    tr3_sym[k,j] = HODLR.HODLR_trace_term(HK, HKj3, opts.saav, false)
  end
  end
  println()
  println("That took $(round(hNtime, 4)) seconds, prepping for block $(+1) out of $(length(hutchN))...")
  println()
end

save("../../data/materntraces.jld", "tr1", tr1_sym, 
                         "tr2", tr2_sym, 
                         "tr3", tr3_sym,
                         "as1", tr1_asm,
                         "as2", tr2_asm,
                         "as3", tr3_asm,
                         "tu1", tapx_tr1,
                         "tu2", tapx_tr2,
                         "tu3", tapx_tr3,
                        )

