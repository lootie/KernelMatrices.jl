
function Base.eltype{T<:Number}(M::KernelMatrix{T})::DataType
  return T
end

function Base.isreal{T<:Number}(M::KernelMatrix{T})::Bool
  return T<:Real
end

function Base.size{T<:Number}(M::KernelMatrix{T})::Tuple{Int64, Int64}
  return length(M.x1), length(M.x2)
end

function Base.size{T<:Number}(M::KernelMatrix{T}, j::Int64)::Int64
  return size(M)[j]
end

function Base.getindex{T<:Number}(M::KernelMatrix{T}, i::Int, j::Int)::T
  return M.kernel(M.x1[i], M.x2[j], M.parms)
end

function getKernelMatrixblock{T<:Number}(M::KernelMatrix{T}, startj::Int, 
                                         endj::Int, startk::Int, endk::Int)::Matrix{T}
  Out = Array{T}(endj-startj+1, endk-startk+1)
    @inbounds begin
    for j in startj:endj
      for k in startk:endk
         Out[j-startj+1,k-startk+1] = M[j,k]
      end
    end
    end
  return Out
end

function Base.full{T<:Number}(M::KernelMatrix{T})::Matrix{T}
  n, m = size(M)
  return getKernelMatrixblock(M, 1, n, 1, m)
end

function Base.getindex{T<:Number}(M::KernelMatrix{T}, jr::UnitRange{Int64}, 
                                  kr::UnitRange{Int64})::Matrix{T}
  return getKernelMatrixblock(M, jr[1], jr[end], kr[1], kr[end])
end

function Base.getindex{T<:Number}(M::KernelMatrix{T}, j::Int64, kr::UnitRange{Int64})::Vector{T}
  Out = Array{T}(length(kr))
  IterTools.@itr for (k, kpt) in enumerate(kr)
     @inbounds Out[k] = M[j, kpt]
  end
  return Out
end

function Base.getindex{T<:Number}(M::KernelMatrix{T}, jr::UnitRange{Int64}, k::Int64)::Vector{T}
  Out = Array{T}(length(jr))
  IterTools.@itr for (j, jpt) in enumerate(jr)
     @inbounds Out[j] = M[jpt, k]
  end
  return Out
end

function Base.getindex{T<:Number}(M::KernelMatrix{T}, j::Int64, ::Colon)::Vector{T}
  kr  = 1:size(M)[2]
  xj  = M.x1[j]
  out = Array{T}(length(kr))
  for j in eachindex(out)
    @inbounds out[j] = M.kernel(xj, M.x2[j], M.parms)
  end
  return out
end

function Base.getindex{T<:Number}(M::KernelMatrix{T}, ::Colon, k::Int64)::Vector{T}
  kr  = 1:size(M)[1]
  xk  = M.x2[k]
  out = Array{T}(length(kr))
  for k in eachindex(out)
    @inbounds out[k] = M.kernel(M.x1[k], xk, M.parms)
  end
  return out
end

function Base.A_mul_B!{T<:Number}(dest::StridedVector, M::KernelMatrix{T}, src::StridedVector)
  sl              = length(src)
  dl              = length(dest)
  sl == size(M)[2] || error("Matrix and Vector sizes do not agree.")
  dl == size(M)[1] || error("Destination and matrix sizes do not agree.")
  roww = Array{eltype(src)}(sl)
  for j in eachindex(dest)
    fillrow!(roww, M, j)
    @inbounds dest[j] = dot(roww, src)
  end
  return dest
end

function Base.At_mul_B!{T<:Number}(dest::StridedVector, M::KernelMatrix{T}, src::StridedVector)
  sl              = length(src)
  dl              = length(dest)
  sl == size(M)[1] || error("Matrix and Vector sizes do not agree.")
  dl == size(M)[2] || error("Destination and matrix sizes do not agree.")
  coll = Array{eltype(src)}(sl)
  for j in eachindex(dest)
    fillcol!(coll, M, j)
    @inbounds dest[j] = dot(coll, src)
  end
  return dest
end

function Base.Ac_mul_B!{T<:Number}(dest::StridedVector, M::KernelMatrix{T}, src::StridedVector)
  sl              = length(src)
  dl              = length(dest)
  sl == size(M)[1] || error("Matrix and Vector sizes do not agree.")
  dl == size(M)[2] || error("Destination and matrix sizes do not agree.")
  coll = Array{eltype(src)}(sl)
  for j in eachindex(dest)
    fillcol!(coll, M, j)
    @inbounds dest[j] = dot(conj(coll), src)
  end
  return dest
end

function Base.A_mul_B!{T<:Number}(dest::StridedMatrix, M::KernelMatrix{T}, src::StridedMatrix)
  size(dest) == size(src) || error("Your target and destination sources don't agree in size")
  nrow = size(M, 1)
  ncol = size(src, 2)
  roww = Array{eltype(M)}(size(M, 2))
  for j in 1:nrow
    fillrow!(roww, M, j)
    for k in 1:ncol
      @inbounds dest[j,k] = dot(roww, src[:,k])
    end
  end
  return dest
end

function Base.At_mul_B!{T<:Number}(dest::StridedMatrix, M::KernelMatrix{T}, src::StridedMatrix)
  size(dest) == size(src) || error("Your target and destination sources don't agree in size")
  mncol = size(M, 2)
  ncol  = size(src, 2)
  coll = Array{eltype(M)}(size(M, 1))
  for j in 1:mncol
    fillcol!(coll, M, j)
    for k in 1:ncol
      @inbounds dest[j,k] = dot(coll, src[:,k])
    end
  end
  return dest
end

function Base.Ac_mul_B!{T<:Number}(dest::StridedMatrix, M::KernelMatrix{T}, src::StridedMatrix)
  size(dest) == size(src) || error("Your target and destination sources don't agree in size")
  mncol = size(M, 2)
  ncol  = size(src, 2)
  coll = Array{eltype(M)}(size(M, 1))
  for j in 1:mncol
    fillcol!(coll, M, j)
    conj!(coll)
    for k in 1:ncol
      @inbounds dest[j,k] = dot(coll, src[:,k])
    end
  end
  return dest
end

function Base.:*{T<:Number}(M::KernelMatrix{T}, x::Vector{T})::Vector{T}
  out = Array{eltype(x)}(length(x))
  A_mul_B!(out, M, x)
  return out
end

function Base.:*{T<:Number}(M::KernelMatrix{T}, x::Matrix{T})::Matrix{T}
  out = Array{eltype(x)}(size(x))
  A_mul_B!(out, M, x)
  return out
end

function Base.diag{T<:Number}(M::KernelMatrix{T}, simple::Bool=false)::Vector{T}
  msz = minimum(size(M))
  out = Array{T}(msz)
  if simple
    fill!(out, M[1,1])
  else
    for j in 1:msz
      @inbounds out[j] = M[j,j]
    end
  end
  return out
end

function Base.full{T<:Number}(L::IncompleteCholesky{T})::Matrix{T}
  Out = L.L[:,1] * L.L[:,1]'
  for j in 2:size(L.L,2)
    Out += L.L[:,j] * L.L[:,j]'
  end
  return Out
end


