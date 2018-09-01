
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

function fillcol!(target::Vector{T}, M::KernelMatrix{T}, idx::Int64)::Nothing where{T<:Number}
  length(target) == size(M)[1] || error("The lengths here don't agree")
  @simd for j in eachindex(target)
     @inbounds target[j] = M[j, idx]
  end
  nothing
end

function fillrow!(target::Vector{T}, M::KernelMatrix{T}, idx::Int64)::Nothing where{T<:Number}
  length(target) == size(M)[2] || error("The lengths here don't agree")
  @simd for j in eachindex(target)
     @inbounds target[j] = M[idx, j]
  end
  nothing
end

function getsortperm(x::Vector{T}, xsorted::Vector{T})::Vector{Int64} where{T}
  sortperm = zeros(Int64, length(x))
  for j in eachindex(x)
    start = 1
    while xsorted[j] != x[start]
      start += 1
    end
    sortperm[j] = start
  end
  return sortperm
end

function shiftpts(v::Vector, coordmin::Float64=1.0, coordmax::Float64=2.0)::Vector
  sv       = deepcopy(v)
  ptd      = length(v[1])
  cmins    = zeros(ptd)
  cmaxs    = zeros(ptd)
  for j in 1:ptd
    cmins[j] = minimum(map(x->x[j], v))
    cmaxs[j] = maximum(map(x->x[j], v))
  end
  for j in eachindex(v)
    for k in 1:ptd
      sv[j][k] = (sv[j][k] - cmins[k])/(cmaxs[k] - cmins[k]) + 1.0
    end
  end
 return sv
end

function hilbertsort(v::Vector)::Vector
  if typeof(v[1]) == Float64
    return sort(v)
  else
    ptd = length(v[1])
    if unique(length.(v)) != [ptd] || !issubset(ptd, [2,3])
      error("Can only handle points in 2- or 3-dimensional space at the moment.")
    end
    shfd = shiftpts(v)
    pts  = map(x->GeometricalPredicates.Point(x...), shfd)
    ptss = deepcopy(pts)
    GeometricalPredicates.hilbertsort!(ptss)
    return v[getsortperm(pts, ptss)]
  end
end

function submatrix(K::KernelMatrix{T}, startj::Int64, stopj::Int64, 
                   startk::Int64, stopk::Int64)::KernelMatrix{T} where{T<:Number}
  return KernelMatrix(view(K.x1, startj:stopj), view(K.x2, startk:stopk), K.parms, K.kernel)
end

function nlfisub(K::KernelMatrix{T}, v::SVector{4, Int64})::KernelMatrix{T} where{T<:Number}
  return submatrix(K, v[1], v[2], v[3], v[4])
end

function submatrix_nystrom(K::KernelMatrix{T}, loj::Int64, hij::Int64, lok::Int64,
                           hik::Int64, landmarkinds::Vector{Int64})::KernelMatrix{T} where{T<:Number}
  K.x1 == K.x2 || error("At least for now, this only works for K.x1 == K.x2")
  nystromkernel = NystromKernel(T, K.kernel, K.x1[landmarkinds], K.parms)
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

