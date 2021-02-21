
# This is the adaptive cross-approximation (ACA), I think originally proposed by
# Bebendorf.  This implementation is an attempt to be as clean and readable as
# possible, but has admittedly grown up a little bit. I probably should split
# some of these things into helper functions.
# 
# TODO (cg 2021/02/21 12:42): this could probably benefit from some more use of
# pre-allocation.
function ACA(M::Union{Matrix{T}, KernelMatrix{T,N,A,Fn}}, rtol::Float64, 
             maxrank::Int64=0)::NTuple{2, Matrix{T}} where{T<:Number,N,A,Fn}

  # Create the maximum size cutoff:
  maxsz     = (maxrank == 0 ? minimum(size(M)) : min(maxrank, minimum(size(M))))
  nro, ncl  = size(M)

  # Declare the vectors for storing cols/rows, keeping track of
  # USED column and row indices, and temporary storage.
  V         = Vector{Vector{T}}(undef, maxsz)
  U         = Vector{Vector{T}}(undef, maxsz)

  # Get the first row in place:
  strt      = 1
  tmprow    = M[strt,:]
  while maximum(abs, tmprow) == 0.0 && strt < maxsz
    strt   += 1
    tmprow  = M[strt,:]
  end
  strt     == maxsz && error("Incompatible maxrank or matrix dimensions.")
  ucol      = Base.BitSet()
  urow      = Base.BitSet(strt)

  # Get the first column in place:
  mv, mi  = KernelMatrices.restrictedmaxabs(tmprow, ucol)
  tmprow./= mv
  tmpcol  = M[:,mi]
  U[1]    = tmpcol
  V[1]    = tmprow
  znorm2  = sum(abs2,tmprow)*sum(abs2,tmpcol)
  push!(ucol, mi)
  frnk = 1

  # Now loop until we have reached the desired tolerance. 
  while norm(tmprow)*norm(tmpcol) > rtol*sqrt(znorm2) && frnk < maxsz
    # Find the next row index:
    mv, mi = KernelMatrices.restrictedmaxabs(tmpcol, urow)
    push!(urow, mi)
    # Get the next row:
    tmprow = M[mi,:]
    for j in 1:frnk
      @inbounds tmprow .-= U[j][mi]*V[j]
    end
    # Find the next column index, use that value to normalize the row, breaking
    # if the value is 0:
    mv, mi = KernelMatrices.restrictedmaxabs(tmprow, ucol)
    mv == 0.0 && break
    push!(ucol, mi)
    tmprow ./= mv
    # Get the next column:
    tmpcol = M[:,mi]
    for j in 1:frnk
      @inbounds tmpcol .-= V[j][mi]*U[j]
    end
    # Update the norm computation:
    for j in 1:frnk
      @inbounds znorm2 += 2.0*abs(dot(U[j], tmpcol))*abs(dot(V[j], tmprow))
    end
    znorm2 += sum(abs2,tmprow)*sum(abs2,tmpcol)
    # Increment the rank by one:
    frnk   += 1
    # Push the new column and row onto the pile:
    V[frnk] = tmprow
    U[frnk] = tmpcol
  end

  # Resize if necessary:
  if frnk < length(U)
    isassigned(U, frnk) && !isassigned(U, frnk+1) || throw(error("ACA array failure."))
    resize!(U, frnk)
    resize!(V, frnk)
  end

  return hcat(U...), hcat(V...)

end

function nystrom_uvt(K::KernelMatrix{T,N,A,Fn}, NK::NystromKernel{T,A}, 
                     plel::Bool)::Tuple{Matrix{T},Matrix{T}} where{T<:Number,N,A,Fn}
  typeof(K.x1[1]) == typeof(NK.lndmk[1]) || error("Nystrom landmarks don't agree with K points.")
  K1  = KernelMatrix(K.x1, NK.lndmk, K.parms, K.kernel)
  K2  = KernelMatrix(NK.lndmk, K.x2, K.parms, K.kernel)
  U   = full(K1, plel)
  V   = transpose(NK.F\full(K2, plel))
  return U,V
end

# TODO (cg 2021/02/21 12:43): these functions don't run anymore because I wrote
# them before 1.0 (!) and don't use them, so never migrated away from the
# A_mul_B syntax. Should do that at some point.
#=
# This is a very simple (and fast!) conversion from an ACA to a partial QR factorization,
# as described in Halko 2011 p238. This is NOT a randomized factorization, and what comes
# out will be numerically equivalent to U*V'.
function QR_ACA(U::Matrix{T}, V::Matrix{T})::Tuple{Matrix{T}, Matrix{T}} where{T<:Number}
  UQ,UR = qr(U)
  D     = A_mul_Bt(UR, V)
  DQ,DR = qr(D)
  Q     = UQ*DQ
  return Q, DR 
end

function SVD_ACA(U::Matrix{T}, V::Matrix{T})::Tuple{Matrix{T}, Vector{T}, Matrix{T}} where{T<:Number}
  UQ,UR    = qr(U)
  D        = A_mul_Bt(UR, V)
  DU,DS,DV = svd(D)
  Uo       = UQ*DU
  return Uo, DS, DV
end
=#


