
Base.eltype(M::KernelMatrix{T})  where{T<:Number} = T
Base.isreal(M::KernelMatrix{T})  where{T<:Number} = T<:Real
Base.size(M::KernelMatrix{T})    where{T<:Number} = (length(M.x1), length(M.x2))
Base.size(M::KernelMatrix{T}, j) where{T<:Number} = size(M)[j]
Base.length(M::KernelMatrix{T})  where{T<:Number} = prod(size(M))

function full(M::KernelMatrix{T}, plel::Bool=false)::Matrix{T} where{T<:Number}
  nworkers() == 1 && return M[:,:]::Matrix{T}
  out = SharedArray{T}(size(M, 1), size(M, 2))
  @sync @distributed for I in CartesianIndices(out)
    out[I] = M.kernel(M.x1[I[1]], M.x2[I[2]], M.parms)
  end
  return collect(out)
end

function Base.getindex(M::KernelMatrix{T}, j::Int64, k::Int64)::T where{T<:Number} 
  return M.kernel(M.x1[j], M.x2[k], M.parms)::Float64
end

function Base.getindex(M::KernelMatrix{T}, j, k)::Array{T} where{T<:Number}
  return [M.kernel(x, y, M.parms) for x in view(M.x1, j), y in view(M.x2, k)]
end

function mul!(dest::StridedVector, M::KernelMatrix{T}, src::StridedVector) where{T<:Number}
  length(src)  == size(M)[2] || error("Matrix and Vector sizes do not agree.")
  length(dest) == size(M)[1] || error("Destination and matrix sizes do not agree.")
  for j in eachindex(dest)
    for k in 1:size(M, 1)
      @inbounds dest[j] += M[j,k]*src[k]
    end
  end
  return dest
end

function mul!(dest::StridedMatrix, M::KernelMatrix{T}, src::StridedMatrix) where{T<:Number}
  size(dest) == size(src) || error("Your target and destination sources don't agree in size")
  row = Array{eltype(M)}(undef, size(M, 2))
  for j in 1:size(M, 1)
    fillrow!(row, M, j)
    for k in 1:size(src, 2)
      @inbounds dest[j,k] = dot(row, src[:,k])
    end
  end
  return dest
end

function LinearAlgebra.:*(M::KernelMatrix{T}, x::Vector{T})::Vector{T} where{T<:Number}
  out = Array{eltype(x)}(undef, length(x))
  return mul!(out, M, x)
end

function LinearAlgebra.:*(M::KernelMatrix{T}, x::Matrix{T})::Matrix{T} where{T<:Number}
  out = Array{eltype(x)}(undef, size(x))
  return mul!(out, M, x)
end

