
Base.eltype(M::KernelMatrix{T,N,A,Fn})  where{T,N,A,Fn} = T
Base.isreal(M::KernelMatrix{T,N,A,Fn})  where{T,N,A,Fn} = T<:Real
Base.size(M::KernelMatrix{T,N,A,Fn})    where{T,N,A,Fn} = (length(M.x1), length(M.x2))
Base.size(M::KernelMatrix{T,N,A,Fn}, j) where{T,N,A,Fn} = size(M)[j]
Base.length(M::KernelMatrix{T,N,A,Fn})  where{T,N,A,Fn} = prod(size(M))

# TODO (cg 2021/02/21 12:41): for this and the method below, perhaps add an
# option with pre-allocated output buffers.
function Base.getindex(M::KernelMatrix{T,0,A,Fn}, j, k)::Array{T} where{T,A,Fn}
  return [M.kernel(x, y) for x in view(M.x1, j), y in view(M.x2, k)]
end

function Base.getindex(M::KernelMatrix{T,N,A,Fn}, j, k)::Array{T} where{T,N,A,Fn}
  return [M.kernel(x, y, M.parms) for x in view(M.x1, j), y in view(M.x2, k)]
end

function Base.getindex(M::KernelMatrix{T,0,A,Fn}, j::Int64, k::Int64)::T where{T,A,Fn} 
  return M.kernel(M.x1[j], M.x2[k])
end

function Base.getindex(M::KernelMatrix{T,N,A,Fn}, j::Int64, k::Int64)::T where{T,N,A,Fn} 
  return M.kernel(M.x1[j], M.x2[k], M.parms)
end

# TODO (cg 2021/02/21 12:40): a specialized method for symmetric M that only
# builds the lower or upper triangle and then returns Symmetric(...).
function full(M::KernelMatrix{T,N,A,Fn}, plel::Bool=false)::Matrix{T} where{T,N,A,Fn}
  !plel && return M[:,:]
  out = Matrix{T}(undef, size(M, 1), size(M, 2))
  Threads.@threads for k in 1:size(M,2)
    @simd for j in 1:size(M,1)
      @inbounds out[j,k] = M[j,k]
    end
  end
  return out
end

function mul!(dest::StridedVector, M::KernelMatrix{T,N,A,Fn}, src::StridedVector) where{T<:Number,N,A,Fn}
  length(src)  == size(M)[2] || error("Matrix and Vector sizes do not agree.")
  length(dest) == size(M)[1] || error("Destination and matrix sizes do not agree.")
  for j in eachindex(dest)
    for k in 1:size(M, 1)
      @inbounds dest[j] += M[j,k]*src[k]
    end
  end
  return dest
end

function mul!(dest::StridedMatrix, M::KernelMatrix{T,N,A,Fn}, src::StridedMatrix) where{T<:Number,N,A,Fn}
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

function LinearAlgebra.:*(M::KernelMatrix{T,N,A,Fn}, x::Vector{T})::Vector{T} where{T<:Number,N,A,Fn}
  out = Array{eltype(x)}(undef, length(x))
  return mul!(out, M, x)
end

function LinearAlgebra.:*(M::KernelMatrix{T,N,A,Fn}, x::Matrix{T})::Matrix{T} where{T<:Number,N,A,Fn}
  out = Array{eltype(x)}(undef, size(x))
  return mul!(out, M, x)
end

