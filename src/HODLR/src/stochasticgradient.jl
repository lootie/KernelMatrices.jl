
function HODLR_trace_apply(HK::KernelHODLR{T}, HKj::DerivativeHODLR{T}, vec::Vector{T})::T where{T<:Number}
  tmp1 = Array{T}(undef, length(vec))
  tmp2 = Array{T}(undef, length(vec))
  _At_ldiv_B!(tmp1, HK.W, vec)
  mul!(  tmp2, HKj,  tmp1)
  return dot(tmp1, tmp2)
end

# Since we are computing the exact HODLR derivative, there are actually not too many tweakable
# options we have to think about. This is a simple call.
function HODLR_grad_term(K::KernelMatrix{T}, HK::KernelHODLR{T}, dfn::Function, dat::Vector{T}, 
                         vecs::Vector{Vector{T}}; plel::Bool=false, verbose::Bool=false) where{T<:Number}
  # Get the EXACT HODLR derivative:
  verbose && println("Assembling derivative matrix...")
  HKj       = DerivativeHODLR(K, dfn, HK, plel=plel)
  # Sequentially apply the solves/products:
  verbose && println("Applying the sequential solves/products...")
  prvec     = HK\(HKj*(HK\dat))
  # Compute the trace term:
  trtm      = mapreduce(v->HODLR_trace_apply(HK, HKj, v), +, vecs)/length(vecs)
  return 0.5*(trtm - dot(dat, prvec))
end

# You need to supply a vector of derivative functions here, which is maybe a little wacky.
# But otherwise, pretty straightforward function.
function stoch_gradient(K::KernelMatrix{T}, HK::KernelHODLR{T}, dat::Vector{T}, dfuns::Vector{Function}, 
                        vecs::Vector{Vector{T}}; plel::Bool=false, verbose::Bool=false, 
                        shuffle::Bool=false)::Vector{T} where{T<:Number}
  # Test stuff:
  HK.U == nothing                  || error("The matrix needs to be factorized for this to work.")
  length(dfuns) == length(K.parms) || error("You didn't supply the right number of gradient funs.")
  # Shuffle the SAA vectors if requested:
  if shuffle
    saa_shuffle!(opts.saav)
  end
  # Loop over the functions:
  out = map(df -> HODLR_grad_term(K, HK, df, dat, vecs, plel=plel, verbose=verbose), dfuns)
  return out
end

# This is the gradient term for the PROFILE log likelihood.
function HODLR_p_grad_term(K::KernelMatrix{T}, HK::KernelHODLR{T}, dfn::Function, dat::Vector{T},
                           vecs::Vector{Vector{T}}; plel::Bool=false, verbose::Bool=false, 
                           seed::Int64=0) where{T<:Number}
  # Get the EXACT HODLR derivative:
  verbose && println("Assembling derivative matrix...")
  HKj       = DerivativeHODLR(K, dfn, HK, plel=plel)
  # Sequentially apply the solves/products:
  verbose && println("Applying the sequential solves/products...")
  prvec     = HK\(HKj*(HK\dat))
  # Compute the trace term:
  verbose && println("Performing stochastic trace estimation...")
  trtm      = mapreduce(v->HODLR_trace_apply(HK, HKj, v), +, vecs)/length(vecs)
  # Compute the output:
  out       = trtm - length(dat)*dot(dat, prvec)/dot(dat, HK\dat)
  return 0.5*out
end

# You need to supply a vector of derivative functions here, which is maybe a little wacky.
# But otherwise, pretty straightforward function.
function stoch_profile_gradient(K::KernelMatrix{T}, HK::KernelHODLR{T}, dat::Vector{T},
                                dfuns::Vector{Function}, vecs::Vector{Vector{T}};
                                plel::Bool=false, verbose::Bool=false,
                                shuffle::Bool=false)::Vector{T} where{T<:Number}
  # Test stuff:
  HK.U == nothing                  || error("The matrix needs to be factorized for this to work.")
  length(dfuns) == length(K.parms) || error("You didn't supply the right number of gradient funs.")
  # Shuffle the SAA vectors if requested:
  if shuffle
    saa_shuffle!(opts.saav)
  end
  # Loop over the functions:
  out = map(df -> HODLR_p_grad_term(K, HK, df, dat, vecs, plel=plel, verbose=verbose), dfuns)
  return out
end

