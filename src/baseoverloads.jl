
function Base.eltype(M::KernelMatrix{T})::DataType where{T<:Number}
  return T
end

function Base.isreal(M::KernelMatrix{T})::Bool where{T<:Number}
  return T<:Real
end

function Base.size(M::KernelMatrix{T})::Tuple{Int64, Int64} where{T<:Number}
  return length(M.x1), length(M.x2)
end

function Base.size(M::KernelMatrix{T}, j::Int64)::Int64 where{T<:Number}
  return size(M)[j]
end

function Base.getindex(M::KernelMatrix{T}, i::Int, j::Int)::T where{T<:Number}
  return M.kernel(M.x1[i], M.x2[j], M.parms)
end

function getKernelMatrixblock(M::KernelMatrix{T}, startj::Int, endj::Int,
                              startk::Int, endk::Int)::Matrix{T} where{T<:Number}
  Out = Array{T}(undef, endj-startj+1, endk-startk+1)
    @inbounds begin
    for j in startj:endj
      for k in startk:endk
         Out[j-startj+1,k-startk+1] = M[j,k]
      end
    end
    end
  return Out
end

function Base.full(M::KernelMatrix{T})::Matrix{T} where{T<:Number}
  n, m = size(M)
  return getKernelMatrixblock(M, 1, n, 1, m)
end

function Base.getindex(M::KernelMatrix{T}, jr::UnitRange{Int64}, 
                       kr::UnitRange{Int64})::Matrix{T} where{T<:Number}
  return getKernelMatrixblock(M, jr[1], jr[end], kr[1], kr[end])
end

function Base.getindex(M::KernelMatrix{T}, j::Int64, kr::UnitRange{Int64})::Vector{T} where{T<:Number}
  Out = Array{T}(undef, length(kr))
  for (k, kpt) in enumerate(kr)
     @inbounds Out[k] = M[j, kpt]
  end
  return Out
end

function Base.getindex(M::KernelMatrix{T}, jr::UnitRange{Int64}, k::Int64)::Vector{T} where{T<:Number}
  Out = Array{T}(undef, length(jr))
  for (j, jpt) in enumerate(jr)
     @inbounds Out[j] = M[jpt, k]
  end
  return Out
end

function Base.getindex(M::KernelMatrix{T}, j::Int64, ::Colon)::Vector{T} where{T<:Number}
  kr  = 1:size(M)[2]
  xj  = M.x1[j]
  out = Array{T}(undef, length(kr))
  for j in eachindex(out)
    @inbounds out[j] = M.kernel(xj, M.x2[j], M.parms)
  end
  return out
end

function Base.getindex(M::KernelMatrix{T}, ::Colon, k::Int64)::Vector{T} where{T<:Number}
  kr  = 1:size(M)[1]
  xk  = M.x2[k]
  out = Array{T}(undef, length(kr))
  for k in eachindex(out)
    @inbounds out[k] = M.kernel(M.x1[k], xk, M.parms)
  end
  return out
end

function LinearAlgebra.A_mul_B!(dest::StridedVector, M::KernelMatrix{T}, src::StridedVector) where{T<:Number}
  sl              = length(src)
  dl              = length(dest)
  sl == size(M)[2] || error("Matrix and Vector sizes do not agree.")
  dl == size(M)[1] || error("Destination and matrix sizes do not agree.")
  roww = Array{eltype(src)}(undef, sl)
  for j in eachindex(dest)
    fillrow!(roww, M, j)
    @inbounds dest[j] = dot(roww, src)
  end
  return dest
end

function LinearAlgebra.At_mul_B!(dest::StridedVector, M::KernelMatrix{T}, src::StridedVector) where{T<:Number}
  sl              = length(src)
  dl              = length(dest)
  sl == size(M)[1] || error("Matrix and Vector sizes do not agree.")
  dl == size(M)[2] || error("Destination and matrix sizes do not agree.")
  coll = Array{eltype(src)}(undef, sl)
  for j in eachindex(dest)
    fillcol!(coll, M, j)
    @inbounds dest[j] = dot(coll, src)
  end
  return dest
end

function LinearAlgebra.Ac_mul_B!(dest::StridedVector, M::KernelMatrix{T}, src::StridedVector) where{T<:Number}
  sl              = length(src)
  dl              = length(dest)
  sl == size(M)[1] || error("Matrix and Vector sizes do not agree.")
  dl == size(M)[2] || error("Destination and matrix sizes do not agree.")
  coll = Array{eltype(src)}(undef, sl)
  for j in eachindex(dest)
    fillcol!(coll, M, j)
    @inbounds dest[j] = dot(conj(coll), src)
  end
  return dest
end

function LinearAlgebra.A_mul_B!(dest::StridedMatrix, M::KernelMatrix{T}, src::StridedMatrix) where{T<:Number}
  size(dest) == size(src) || error("Your target and destination sources don't agree in size")
  nrow = size(M, 1)
  ncol = size(src, 2)
  roww = Array{eltype(M)}(undef, size(M, 2))
  for j in 1:nrow
    fillrow!(roww, M, j)
    for k in 1:ncol
      @inbounds dest[j,k] = dot(roww, src[:,k])
    end
  end
  return dest
end

function LinearAlgebra.At_mul_B!(dest::StridedMatrix, M::KernelMatrix{T}, src::StridedMatrix) where{T<:Number}
  size(dest) == size(src) || error("Your target and destination sources don't agree in size")
  mncol = size(M, 2)
  ncol  = size(src, 2)
  coll = Array{eltype(M)}(undef, size(M, 1))
  for j in 1:mncol
    fillcol!(coll, M, j)
    for k in 1:ncol
      @inbounds dest[j,k] = dot(coll, src[:,k])
    end
  end
  return dest
end

function LinearAlgebra.Ac_mul_B!(dest::StridedMatrix, M::KernelMatrix{T}, src::StridedMatrix) where{T<:Number}
  size(dest) == size(src) || error("Your target and destination sources don't agree in size")
  mncol = size(M, 2)
  ncol  = size(src, 2)
  coll = Array{eltype(M)}(undef, size(M, 1))
  for j in 1:mncol
    fillcol!(coll, M, j)
    conj!(coll)
    for k in 1:ncol
      @inbounds dest[j,k] = dot(coll, src[:,k])
    end
  end
  return dest
end

function LinearAlgebra.:*(M::KernelMatrix{T}, x::Vector{T})::Vector{T} where{T<:Number}
  out = Array{eltype(x)}(undef, length(x))
  A_mul_B!(out, M, x)
  return out
end

function LinearAlgebra.:*(M::KernelMatrix{T}, x::Matrix{T})::Matrix{T} where{T<:Number}
  out = Array{eltype(x)}(undef, size(x))
  A_mul_B!(out, M, x)
  return out
end

function LinearAlgebra.diag(M::KernelMatrix{T}, simple::Bool=false)::Vector{T} where{T<:Number}
  msz = minimum(size(M))
  out = Array{T}(undef, msz)
  if simple
    fill!(out, M[1,1])
  else
    for j in 1:msz
      @inbounds out[j] = M[j,j]
    end
  end
  return out
end

function Base.full(L::IncompleteCholesky{T})::Matrix{T} where{T<:Number}
  Out = L.L[:,1] * L.L[:,1]'
  for j in 2:size(L.L,2)
    Out += L.L[:,j] * L.L[:,j]'
  end
  return Out
end


