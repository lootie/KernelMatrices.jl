
# This is the big kahuna function right here. I have liberally commented it and
# written some helper functions with long specific names in the hopes that this
# is human-readable. Some of the helper functions are uglier and less easy to
# read, but it is my hope that this chunk of code is intelligible.
function symmetricfactorize!(K::KernelHODLR{T}; plel::Bool=false, 
                             verbose::Bool=false) where{T<:Number}

  # If parallel option is given, warn about BLAS.set_num_threads():
  if plel && verbose
    @info "Without BLAS.set_num_threads(1), you may not see much multi-threaded speedup." maxlog=1
  end

  # Get the cholfact of each of the leaves, so that K.L[j] = W[j]*W[j]'. Then,
  # get the inverse of each of those symmetric factors.
  verbose && println("Computing factors for leaves...")
  LW = mapf(x->factorize(cholesky(x).L), K.L, plel)

  # Now, apply the leaf inverses blockwise to each of the U factors at all levels.
  verbose && println("Applying factor inverse to each non-leaf...")
  foreachf(lv->invapply!(LW, K.U[lv], K.V[lv]), eachindex(K.U), plel)

  # Declare the array for all the non-leaf symmetric factor terms:
  nonleafW = Vector{Vector{LowRankW{T}}}(undef, length(K.U))

  # Now we loop down each level, using the low rank symmetric factorization
  # to get symmetric factors and then the Woodbury formula to apply it to all levels.
  for lv in 1:length(K.U)
    # Get the low-rank symmetric factors for this level.
    verbose && println("Factoring level $lv of non-leaves...")
    nonleafW[lv] = mapf(x->lrsymfact(x[1], x[2]), zip(K.V[lv], K.U[lv]), plel)
    # Apply their inverse to all the lower levels.
    verbose && println("Applying level $lv of non-leaf inverses to each non-leaf...")
    levs = (lv+1):length(K.U)
    if length(levs) > 0
      foreachf(lev->invapply!(nonleafW[lv], K.U[lev], K.V[lev]), levs, plel)
    end
  end
  
  # Put it in place:
  K.W = FactorHODLR{T}(LW, nonleafW)

  # Ditch the U, V, as they have been overwritten and are no longer helpful.
  verbose && @warn("U and V abandoned after overwriting")
  K.U = nothing
  K.V = nothing

  nothing
end

