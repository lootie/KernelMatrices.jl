
function restrictedmaxabs(x::Vector{T}, noinds::BitSet)::Tuple{T, Int64} where{T<:Number}
  ind = 1
  while in(ind, noinds) && ind < length(x)
    ind += 1
  end
  val = 0.0
  @simd for j in ind:length(x)
     @inbounds if abs(x[j]) > abs(val) && !in(j, noinds)
      ind = j
      val = x[j]
    end
  end
  return val, ind
end

function fillcol!(target::Vector{T}, M::KernelMatrix{T,N,A,Fn}, idx::Int64)::Nothing where{T<:Number,N,A,Fn}
  length(target) == size(M)[1] || error("The lengths here don't agree")
  @simd for j in eachindex(target)
     @inbounds target[j] = M[j, idx]
  end
  nothing
end

function fillrow!(target::Vector{T}, M::KernelMatrix{T,N,A,Fn}, idx::Int64)::Nothing where{T<:Number,N,A,Fn}
  length(target) == size(M)[2] || error("The lengths here don't agree")
  @simd for j in eachindex(target)
     @inbounds target[j] = M[idx, j]
  end
  nothing
end

function getsortperm(x, xsorted)::Vector{Int64}
  D = Dict(zip(x, eachindex(x)))
  return [D[xj] for xj in xsorted]
end

function data_reorder(data, x, xsorted)
  D   = Dict(zip(x, data))
  return [D[xj] for xj in xsorted]
end

function submatrix(K::KernelMatrix{T,N,A,Fn}, startj::Int64, stopj::Int64, 
                   startk::Int64, stopk::Int64)::KernelMatrix{T,N,A,Fn} where{T<:Number,N,A,Fn}
  return KernelMatrix(view(K.x1, startj:stopj), view(K.x2, startk:stopk), K.parms, K.kernel)
end

function nlfisub(K::KernelMatrix{T,N,A,Fn}, v::SVector{4, Int64})::KernelMatrix{T,N,A,Fn} where{T<:Number,N,A,Fn}
  return submatrix(K, v[1], v[2], v[3], v[4])
end

function submatrix_nystrom(K::KernelMatrix{T,N,A,Fn}, loj::Int64, hij::Int64, lok::Int64,
                           hik::Int64, landmarkinds::Vector{Int64})::KernelMatrix{T,N,A,Fn} where{T<:Number,N,A,Fn}
  K.x1 == K.x2 || error("At least for now, this only works for K.x1 == K.x2")
  nystromkernel = NystromKernel(K.kernel, K.x1[landmarkinds], K.parms)
  return KernelMatrix(view(K.x1, loj:hij), view(K.x2, lok:hik), K.parms, nystromkernel)
end

function fill_landmarks!(lmk_mutate::AbstractVector, xpts::AbstractVector)
  inds = Int64.(round.(LinRange(1, length(xpts), length(lmk_mutate))))
  @simd for j in eachindex(inds)
    @inbounds lmk_mutate[j] = xpts[inds[j]]
  end
  nothing
end

function vv_to_m(V::Vector{Vector{T}})::Matrix{T} where{T}
  V1len = length(V[1])
  Out   = zeros(T, V1len, length(V))
  for j in eachindex(V)
    @simd for k in 1:size(Out, 1)
      @inbounds Out[k,j] = V[j][k]
    end
  end
  return Out
end

