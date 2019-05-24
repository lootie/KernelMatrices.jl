
# This is the big kahuna function right here. I have liberally commented it and written some helper
# functions with long specific names in the hopes that this is human-readable. Some of the helper
# functions are uglier and less easy to read, but it is my hope that this chunk of code is intelligible.
function symmetricfactorize!(K::KernelHODLR{T}; plel::Bool=false, verbose::Bool=false) where{T<:Number}

  # Get the numbers of workers once:
  nwrk  = nworkers()

  # Get the cholfact of each of the leaves, so that K.L[j] = W[j]*W[j]'. Then,
  # get the inverse of each of those symmetric factors.
  verbose && println("Computing factors for leaves...")
  LW    = mapf(symfact, K.L,             nwrk, plel)
  LWf   = mapf(lu,  LW,                  nwrk, plel)
  LWtf  = mapf(lu,  imap(transpose, LW), nwrk, plel)

  # Now, apply the leaf inverses blockwise to each of the U factors at all levels.
  verbose && println("Applying factor inverse to each non-leaf...")
  for lv in eachindex(K.U)
    invapply!(LWf, lv, K.U, K.V, false)
  end

  # Declare the array for all the non-leaf symmetric factor terms:
  nonleafW = Vector{Vector{LowRankW{T}}}(undef, length(K.U))

  # Now we loop down each level, using the low rank symmetric factorization
  # to get symmetric factors and then the Woodbury formula to apply it to all levels.
  for lv in 1:length(K.U)
    # Get the low-rank symmetric factors for this level.
    verbose && println("Factoring level $lv of non-leaves...")
    tmpW   = mapf(x->lrsymfact(x[1], x[2]), zip(K.V[lv], K.U[lv]), nwrk, plel)
    # Apply their inverse to all the lower levels.
    verbose && println("Applying level $lv of non-leaf inverses to each non-leaf...")
    for lev in (lv+1):length(K.U)
      invapply!(tmpW, lev, K.U, K.V, false)
    end
    # Push tmpW onto nonleafW.
    nonleafW[lv] = tmpW
  end
  
  # Put it in place:
  K.W = FactorHODLR{T}(LW, LWf, LWtf, nonleafW)

  # Ditch the U, V, as they have been overwritten and are no longer helpful.
  verbose && @warn("U and V abandoned after overwriting")
  K.U = nothing
  K.V = nothing

  nothing
end

