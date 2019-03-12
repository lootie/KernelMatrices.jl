
function Base.size(K::KernelHODLR{T})::Tuple{Int64, Int64} where{T<:Number}
  sz = sum(map(x->size(x,1), K.L))
  return sz, sz
end

function Base.size(K::KernelHODLR{T}, idx::Int64)::Int64 where{T<:Number}
  abs(idx) <= 2 || error("2D array-like thing.")
  sz = sum(map(x->size(x,1), K.L))
  return sz
end

function Base.size(W::LowRankW{T})::Tuple{Int64, Int64} where{T<:Number}
  return size(W.M,1), size(W.M,1)
end

function Base.size(W::LowRankW{T}, j::Int64)::Int64 where{T<:Number}
  return ifelse(j==1, size(W.M,1), size(W.M,2))
end

function full(M::LowRankW{T})::Matrix{T} where{T<:Number}
  return I + mul_t(M.M*M.X, M.M)
end

function LinearAlgebra.det(W::LowRankW{T})::Float64 where{T<:Number}
  return det(I + t_mul(W.M, W.M)*W.X)
end

function full(K::KernelHODLR{T})::Matrix{T} where{T<:Number}
  Out = Array{T}(undef, size(K))
  for (j,pt) in enumerate(K.leafindices)
    Out[pt[1]:pt[2], pt[3]:pt[4]] = K.L[j]
  end
  for lev in eachindex(K.U)
    for j in eachindex(K.U[lev])
      a, b, c, d    = K.nonleafindices[lev][j]
      Out[a:b, c:d] = mul_t(K.U[lev][j], K.V[lev][j])
      Out[c:d, a:b] = mul_t(K.V[lev][j], K.U[lev][j])
    end
  end
  return Out
end

# VERY computationally inefficient. This really is only for testing.
function full(W::FactorHODLR{T})::Matrix{T} where{T<:Number}
  Out = cat(W.leafW..., dims=[1,2])
  # Multiply the nonleaves:
  for j in 1:length(W.nonleafW)
    Out = Out*cat(full.(W.nonleafW[j])..., dims=[1,2])
  end
  return Out
end

function LinearAlgebra.mul!(target::StridedArray, W::LowRankW{T}, src::StridedArray) where{T<:Number}
  # Zero out target:
  fill!(target, zero(eltype(target)))  
  # Do multiplication:
  mul!(target, W.M, W.X*t_mul(W.M, src))
  @simd for j in eachindex(target)
     @inbounds target[j] += src[j]
  end
  return target
end

function _At_mul_B!(target::StridedArray, W::LowRankW{T}, src::StridedArray) where{T<:Number}
  # Zero out target:
  fill!(target, zero(eltype(target)))  
  # Do multiplication:
  mul!(target, W.M, t_mul(W.X, t_mul(W.M, src)))
  @simd for j in eachindex(target)
     @inbounds target[j] += src[j]
  end
  return target
end

function LinearAlgebra.ldiv!(target::StridedArray, W::LowRankW{T}, src::StridedArray) where{T<:Number}
  # Zero out target:
  fill!(target, zero(eltype(target)))  
  # Do multiplication:
  mul!(target, W.M, -lrx_solterm(W, t_mul(W.M, src)))
  @simd for j in eachindex(target)
     @inbounds target[j] += src[j]
  end
  return target
end

function _At_ldiv_B!(target::StridedArray, W::LowRankW{T}, src::StridedArray) where{T<:Number}
  # Zero out target:
  fill!(target, zero(eltype(target)))  
  # Do multiplication:
  mul!(target, W.M, -lrx_solterm_t(W, t_mul(W.M, src)))
  @simd for j in eachindex(target)
     @inbounds target[j] += src[j]
  end
  return target
end

function LinearAlgebra.mul!(target::StridedVector, W::FactorHODLR{T}, src::StridedVector) where{T<:Number}
  # Zero out the target vector, get tmp vector:
  fill!(target, zero(eltype(target)))
  # Apply the nonleafW vectors in the correct order:
  tmp = deepcopy(src)
  for j in length(W.nonleafW):-1:1
    tmp = apply_block(W.nonleafW[j], tmp, false, false)
  end
  # Apply the leaf vectors:
  tmp = apply_block(W.leafW, tmp, false, false)
  fillall!(target, tmp)
  return target
end

function _At_mul_B!(target::StridedVector, W::FactorHODLR{T}, src::StridedVector) where{T<:Number}
  # Zero out the target vector, get tmp vector:
  fill!(target, zero(eltype(target)))
  # Apply the leaf vectors:
  tmp = deepcopy(src)
  tmp = apply_block(W.leafW, tmp, false, true)
  # Apply the nonleafW vectors in the correct order:
  for j in eachindex(W.nonleafW)
    tmp = apply_block(W.nonleafW[j], tmp, false, true)
  end
  fillall!(target, tmp)
  return target
end

function _At_mul_B!(target::StridedVector, A::StridedMatrix, src::StridedVector)
  mul!(target, transpose(A), src)
end

function LinearAlgebra.ldiv!(target::StridedVector, W::FactorHODLR{T}, src::StridedVector) where{T<:Number}
  # Zero out the target vector, get tmp vector:
  fill!(target, zero(eltype(target)))
  # Apply the leaf vectors:
  tmp = deepcopy(src)
  tmp = apply_block(W.leafWf, tmp, true, false)
  # Apply the nonleafW vectors in the correct order:
  for j in eachindex(W.nonleafW)
    tmp = apply_block(W.nonleafW[j], tmp, true, false)
  end
  fillall!(target, tmp)
  return target
end

function _At_ldiv_B!(target::StridedVector, W::FactorHODLR{T}, src::StridedVector) where{T<:Number}
  # Zero out the target vector, get tmp vector:
  fill!(target, zero(eltype(target)))
  # Apply the nonleafW vectors in the correct order:
  tmp = deepcopy(src)
  for j in length(W.nonleafW):-1:1
    tmp = apply_block(W.nonleafW[j], tmp, true, true)
  end
  # Apply the leaf vectors:
  tmp = apply_block(W.leafWtf, tmp, true, false)
  fillall!(target, tmp)
  return target
end

function LinearAlgebra.mul!(target::StridedVector, K::KernelHODLR{T}, src::StridedVector) where{T<:Number}
  if K.W == nothing
    # Zero out the target vector:
    fill!(target, zero(eltype(target)))
    # Apply the leaves:
    for j in eachindex(K.L)
      c, d = K.leafindices[j][3:4]
      mul!(view(target, c:d), K.L[j], src[c:d])
    end
    # Apply the non-leaves:
    for j in eachindex(K.U)
      for k in eachindex(K.U[j])
        a, b, c, d   = K.nonleafindices[j][k]
        target[c:d] += K.V[j][k]*(transpose(K.U[j][k])*src[a:b])
        target[a:b] += K.U[j][k]*(transpose(K.V[j][k])*src[c:d])
      end
    end
  else
    # Zero out the target vector:
    fill!(target, zero(eltype(target)))
    # Multiple by W^{T}, then by W:
    tmp = Array{eltype(target)}(undef, length(target))
    _At_mul_B!(tmp, K.W, src)
    mul!(target, K.W, tmp)
  end
  return target
end

function _At_mul_B!(target::StridedVector, K::KernelHODLR{T}, src::StridedVector) where{T<:Number}
  return mul!(target, K, src)
end

function LinearAlgebra.ldiv!(target::StridedVector, K::KernelHODLR{T}, src::StridedVector) where{T<:Number}
  if K.W == nothing
    error("No solves without factorization.")
  else
    # divide by W, then by W^{T}:
    tmp = Array{eltype(target)}(undef, length(target))
    ldiv!(tmp, K.W, src)
    _At_ldiv_B!(target, K.W, tmp)
  end
  return target
end

function _At_ldiv_B!(target::StridedVector, K::KernelHODLR{T}, src::StridedVector) where{T<:Number}
  return ldiv!(target, K, src)
end

function LinearAlgebra.logdet(K::KernelHODLR{T})::Float64 where{T<:Number}
  if K.W == nothing
    error("No logdet without factorization.")
  else
    logdett = 0.0
    for j in eachindex(K.L)
      @inbounds logdett += logdet(factorize(K.L[j]))
    end
    for j in eachindex(K.W.nonleafW)
      for k in eachindex(K.W.nonleafW[j])
         @inbounds logdett += 2.0*log(abs(det(K.W.nonleafW[j][k])))
      end
    end
    return logdett
  end
end

function Base.size(DK::DerivativeHODLR{T})::Tuple{Int64, Int64} where{T<:Number}
  sz = sum(map(x->size(x,1), DK.L))
  return sz, sz
end

function full(DK::DerivativeHODLR{T})::Matrix{T} where{T<:Number}
  Out = zeros(T, size(DK))
  for (j,pt) in enumerate(DK.leafindices)
    Out[pt[1]:pt[2], pt[3]:pt[4]] = DK.L[j]
  end
  for lev in eachindex(DK.B)
    for j in eachindex(DK.B[lev])
      a, b, c, d     = DK.nonleafindices[lev][j]
      Out[a:b, c:d]  = DBlock_full(DK.B[lev][j], DK.S, DK.Sj)
      Out[c:d, a:b]  = transpose(Out[a:b, c:d])
    end
  end
  return Out
end

function LinearAlgebra.mul!(target::StridedVector, DK::DerivativeHODLR{T}, src::StridedVector) where{T<:Number}
  # Zero out the target vector:
  fill!(target, zero(eltype(target)))
  # Apply the leaves:
  for j in eachindex(DK.L)
    c, d = DK.leafindices[j][3:4]
    target[c:d] += DK.L[j]*src[c:d]
  end
  # Apply the non-leaves:
  for j in eachindex(DK.B)
    for k in eachindex(DK.B[j])
      a, b, c, d   = DK.nonleafindices[j][k]
      target[c:d] += DBlock_mul_t(DK.B[j][k], src[a:b], DK.S, DK.Sj)
      target[a:b] += DBlock_mul(DK.B[j][k],   src[c:d], DK.S, DK.Sj)
    end
  end
  return target
end

##
#
# Convenience overloads:
#
##

function LinearAlgebra.:*(K::KernelHODLR{T}, source::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, length(source))
  mul!(target, K, source)
  return target
end

function LinearAlgebra.:*(W::FactorHODLR{T}, source::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, length(source))
  mul!(target, W, source)
  return target
end

function LinearAlgebra.:\(K::KernelHODLR{T}, source::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, length(source))
  ldiv!(target, K, source)
  return target
end

function LinearAlgebra.:*(W::LowRankW{T}, src::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, length(source))
  mul!(target, W, src)
  return target
end

function LinearAlgebra.:*(W::LowRankW{T}, src::Matrix{T})::Matrix{T} where{T<:Number}
  target = Array{T}(undef, size(src))
  mul!(target, W, src)
  return target
end

function LinearAlgebra.:*(DK::DerivativeHODLR{T}, src::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, size(src))
  mul!(target, DK, src)
  return target
end

function LinearAlgebra.:\(W::LowRankW{T}, src::Vector{T})::Vector{T} where{T<:Number}
  target = Array{T}(undef, length(source))
  ldiv!(target, W, src)
  return target
end

function LinearAlgebra.:\(W::LowRankW{T}, src::Matrix{T})::Matrix{T} where{T<:Number}
  target = Array{T}(undef, size(src))
  ldiv!(target, W, src)
  return target
end

