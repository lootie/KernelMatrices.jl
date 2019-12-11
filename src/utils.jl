
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

function getsortperm(x, xsorted)::Vector{Int64}
  D = Dict(zip(x, eachindex(x)))
  return [D[xj] for xj in xsorted]
end

function data_reorder(data, x, xsorted)
  D   = Dict(zip(x, data))
  return [D[xj] for xj in xsorted]
end

function submatrix(K::KernelMatrix, startj::Int64, stopj::Int64, 
                   startk::Int64, stopk::Int64)
  return KernelMatrix(view(K.x1, startj:stopj), view(K.x2, startk:stopk), 
                      K.parms, K.kernel)
end

@inline submatrix(K, v) = submatrix(K, v...)

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

