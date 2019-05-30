
# The constructor for the HODLR matrix given a KernelMatrix struct. If you nystrom=false, the blocks
# are assembled with the ACA up to tolerance ep or rank maxrank (if maxrank>0, otherwise no limit on
# the permitted rank). If nystrom=true, assembles blocks using the Nystrom approximation.
function KernelHODLR(K::KernelMatrix{T}, ep::Float64, maxrank::Int64, lvl::HierLevel;
                     nystrom::Bool=false, plel::Bool=false)::KernelHODLR{T} where{T<:Number}

  # Get the level, leaf indices, and non-leaf indices:
  level, leafinds, nonleafinds = HODLRindices(size(K)[1], lvl)
  nwrk                         = nworkers()

  # If the Nystrom method was requested, prepare that:
  if nystrom
    maxrank < 2 && error("Please supply a fixed off-diagonal rank that is ≧2.")
    if maxrank >= minimum(map(x->min(x[2]-x[1], x[4]-x[3]), leafinds))
      error("Your nystrom rank is too big. Reduce the HODLR level or nystrom rank.")
    end
    K.x1 == K.x2 || error("This type of matrix doesn't admit a Nystrom kernel appx. Need x1 == x2")
    nyind = Int64.(round.(LinRange(1, size(K)[1], maxrank)))
    nyker = NystromKernel(T, K.kernel, K.x1[nyind], K.parms, true)
  end

  # Get the leaves in position:
  leaves = mapf(x->Symmetric(full(nlfisub(K, x), plel)), leafinds, nwrk, plel)

  # Get the rest of the decompositions of the non-leaf nodes in place:
  U = Vector{Vector{Matrix{T}}}(undef, level)  
  V = Vector{Vector{Matrix{T}}}(undef, level)  
  for j in eachindex(U)
    if nystrom
      nonleafinds[j]
      tmpUV = mapf(x->nystrom_uvt(nlfisub(K, x), nyker, plel), nonleafinds[j], nwrk, plel)
    else
      tmpUV = mapf(x->ACA(nlfisub(K, x), ep, maxrank), nonleafinds[j], nwrk, plel)
    end
    U[j] = map(x->x[1], tmpUV)
    V[j] = map(x->x[2], tmpUV)
  end

  return KernelHODLR{T}(ep, level, maxrank, leafinds, nonleafinds, U, V, leaves, nothing, nystrom)

end



# The constructor for the EXACT derivative of a HODLR matrix. It doesn't actually require the blocks
# of the HODLR matrix, but passing it the HODLR matrix is convenient to access the information about 
# block boundaries and stuff.
function DerivativeHODLR(K::KernelMatrix{T}, dfun::Function, HK::KernelHODLR{T}; 
                         plel::Bool=false) where{T<:Number}
  # Check that the call is valid:
  HK.nys || error("This is only valid for Nystrom-block matrices.")
  nwrk   = nworkers()

  # Get the landmark point vector, and global S and Sj:
  lndmk  = K.x1[Int64.(round.(LinRange(1, size(K)[1], HK.mrnk)))]
  S      = cholesky(Symmetric(full(KernelMatrix(lndmk, lndmk, K.parms, K.kernel), plel)))
  Sj     = Symmetric(full(KernelMatrix(lndmk, lndmk, K.parms, dfun), plel))

  # Declare the derivative kernel matrix:
  dK     = KernelMatrix(K.x1, K.x2, K.parms, dfun)

  # Get the leaves in position:
  leaves = mapf(x->Symmetric(full(nlfisub(dK, x), plel)), HK.leafindices, nwrk, plel)

  # Get the non-leaves in place:
  B      = Vector{Vector{DerivativeBlock{T}}}(undef, HK.lvl)
  for j in eachindex(B)
    B[j] = mapf(x->DBlock(nlfisub(K, x), dfun, lndmk, plel), HK.nonleafindices[j], nwrk, plel)
  end

  return DerivativeHODLR(HK.ep, HK.lvl, HK.leafindices, HK.nonleafindices, leaves, B, S, Sj)
end



# Construct the leaves of the EXACT second derivative of a HODLR matrix.
function SecondDerivativeLeaves(K::KernelMatrix{T}, djk::Function, lfi::AbstractVector, 
                                plel::Bool=false) where{T<:Number}
  d2K    = KernelMatrix(K.x1, K.x2, K.parms, djk)
  return mapf(x->Symmetric(full(nlfisub(d2K, x), plel)), lfi, nworkers(), plel)
end



# Construct the off-diagonal blocks of the EXACT second derivative of a HODLR matrix.
function SecondDerivativeBlocks(K::KernelMatrix{T}, djk::Function, nlfi::AbstractVector,
                                mrnk::Int64, plel::Bool=false) where{T<:Number}
  d2K    = KernelMatrix(K.x1, K.x2, K.parms, djk)
  lndmk  = K.x1[Int64.(round.(LinRange(1, size(K)[1], mrnk)))]
  B      = map(nlf -> mapf(x->SBlock(nlfisub(d2K, x), djk, lndmk), nlf, nworkers(), plel), nlfi)
  return B
end

