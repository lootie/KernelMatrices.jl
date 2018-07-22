
# Get the leaf sizes. The leaves will have sizes between fix_level and 2*fix_level. If fix_level
# isn't valid, it will be adjusted, sometimes without warning.
function _leafsizes(N::Int64, fix_level::Int64=0, base::Int64=2)::Tuple{Int64, Vector{Int64}}
  n       = Int64(floor(log(base, N)))
  if fix_level > 0 && fix_level <= n
    n     = fix_level
  elseif fix_level > 0 && fix_level > n
    warn("You can't have the level that you requested, dropping down to highest permissible.")
  else
    n     = max(n-7, 1) # This actually corresponds to log2(n) - 8.
  end
  leafszs = base*ones(base^(n-1))
  remandr = N-base^n
  while remandr > 0
    for j in 1:base^(n-1)
      leafszs[j] += 1
      remandr    -= 1
      if remandr == 0
        break
      end
    end
  end
  return n, leafszs
end

# The output of this function will be a vector of vectors with: 
# [xstart, xstop, ystart, ystop],  so that the j-th leaf of the 
# matrix K will be K[xstart:xstop, ystart:ystop].
function _leafindices(szs::Vector{Int64})::Vector{SVector{4, Int64}}
  Out  = SVector{4, Int64}[]
  strt = 1
  for j in eachindex(szs)
    push!(Out, SVector(strt, strt+szs[j]-1, strt, strt+szs[j]-1))
    strt += szs[j]
  end
  return Out
end

# Get the j-th level non-leaf indices given the leaf indices.
function _nextlevel(leafindices::Vector{SVector{4, Int64}}, lvl::Int64)::Vector{SVector{4, Int64}}
  Out   = SVector{4, Int64}[]
  skp   = 2^lvl
  skpd2 = 2^(lvl-1)
  for j in 1:skp:(length(leafindices)-1)
    push!(Out, SVector(leafindices[j+skpd2  ][1], 
                       leafindices[j+skp-1  ][2], 
                       leafindices[j        ][3], 
                       leafindices[j+skpd2-1][4]))
  end
  return Out
end

# Putting it all together, give this function an N and get the
# indices of every HODLR node.
function HODLRindices(N::Int64, kmax::Int64=0)::Tuple{Int64, 
                                                      Vector{SVector{4, Int64}}, 
                                                      Vector{Vector{SVector{4, Int64}}}}
  lv, lfszs = _leafsizes(N, kmax)
  lfind     = _leafindices(lfszs)
  nonlfind  = [_nextlevel(lfind, 1)]
  for j in 2:(lv-1)
    push!(nonlfind, _nextlevel(lfind, j))
  end
  return lv, lfind, nonlfind
end

