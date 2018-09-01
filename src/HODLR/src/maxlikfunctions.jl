
function negloglik(HK::Union{Matrix{T}, KernelHODLR{T}}, dat::Vector{T}) where{T<:Number}
  typeof(HK) == KernelHODLR{T} || warn("This may take a while on a full matrix...")
  if typeof(HK) == KernelHODLR{T} && HK.U != nothing
    error("The HODLR matrix needs to be factorized for this.")
  end
  nll = 0.5*logdet(HK) + 0.5*dot(dat, HK\dat)
  return nll
end

function profile_negloglik(HK::Union{Matrix{T}, KernelHODLR{T}}, dat::Vector{T}) where{T<:Number}
  typeof(HK) == KernelHODLR{T} || warn("This may take a while on a full matrix...")
  if typeof(HK) == KernelHODLR{T} && HK.U != nothing
    error("The HODLR matrix needs to be factorized for this.")
  end
  nll = logdet(HK) + length(dat)*log(dot(dat, HK\dat))
  return 0.5*nll
end

function scaleparm_mle(prms::AbstractVector, locs::AbstractVector, dats::AbstractVector, opts::Maxlikopts)
  nllK = KernelMatrices.KernelMatrix(locs, locs, prms, opts.kernfun)
  HK   = KernelHODLR(nllK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  symmetricfactorize!(HK, plel=opts.fpll)
  scal = dot(dats, HK\dats)/length(dats)
  return scal
end

function nll_objective(prms::AbstractVector, grad::Vector, locs::AbstractVector,
                       dats::AbstractVector, opts::Maxlikopts)
  opts.verb && @show prms
  nllK = KernelMatrices.KernelMatrix(locs, locs, prms, opts.kernfun)
  tim1 = @elapsed begin
  HK   = KernelHODLR(nllK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  symmetricfactorize!(HK, plel=opts.fpll)
  end
  opts.verb && println("Assembly+factorize took       $(round(tim1, 3)) seconds.")
  tim2 = @elapsed nll     = negloglik(HK, dats)
  opts.verb && println("Negative log-lik took         $(round(tim2, 3)) seconds.")
  tim3 = @elapsed begin
  if length(grad) > 0
    grad .= stoch_gradient(nllK, HK, dats, opts.dfuns, opts.saav, plel=opts.apll, shuffle=!(opts.saa_fix))
  end
  end
  opts.verb && println("Gradient took                 $(round(tim3, 3)) seconds.")
  opts.verb && println()
  opts.verb && println("All told, this objective function call took $(round(tim1+tim2+tim3, 3)) seconds.")
  opts.verb && println()
  return nll
end

function nlpl_scale(prms::AbstractVector, locs::AbstractVector, dats::AbstractVector, opts::Maxlikopts)
  nllK = KernelMatrices.KernelMatrix(locs, locs, prms, opts.kernfun)
  HK   = KernelHODLR(nllK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  symmetricfactorize!(HK, plel=opts.fpll)
  return dot(dats, HK\dats)/length(dats)
end

# Negative Log PROFILE likelihood
function nlpl_objective(prms::AbstractVector, grad::Vector, locs::AbstractVector,
                       dats::AbstractVector, opts::Maxlikopts)
  opts.verb && @show prms
  nllK = KernelMatrices.KernelMatrix(locs, locs, prms, opts.kernfun)
  tim1 = @elapsed begin
  HK   = KernelHODLR(nllK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  symmetricfactorize!(HK, plel=opts.fpll)
  end
  opts.verb && println("Assembly+factorize took       $(round(tim1, 3)) seconds.")
  tim2 = @elapsed nll     = profile_negloglik(HK, dats)
  opts.verb && println("Negative log-lik took         $(round(tim2, 3)) seconds.")
  tim3 = @elapsed begin
  if length(grad) > 0
    grad .= stoch_profile_gradient(nllK, HK, dats, opts.dfuns, opts.saav, plel=opts.apll, shuffle=!(opts.saa_fix))
  end
  end
  opts.verb && println("Gradient took                 $(round(tim3, 3)) seconds.")
  opts.verb && println()
  opts.verb && println("All told, this objective function call took $(round(tim1+tim2+tim3, 3)) seconds.")
  opts.verb && println()
  return nll
end

function nll_gradient(prms::AbstractVector, locs::AbstractVector, dats::AbstractVector, opts::Maxlikopts)
  nllK = KernelMatrices.KernelMatrix(locs, locs, prms, opts.kernfun)
  tim1 = @elapsed begin
  HK   = KernelHODLR(nllK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  symmetricfactorize!(HK, plel=opts.fpll)
  end
  opts.verb && println("Assembly+factorize took       $(round(tim1, 3)) seconds.")
  tim2 = @elapsed begin
  grad = stoch_gradient(nllK, HK, dats, opts.dfuns, opts.saav, plel=opts.apll, shuffle=!(opts.saa_fix))
  end
  opts.verb && println("Gradient took                 $(round(tim2, 3)) seconds.")
  opts.verb && println()
  return grad
end

function nll_hessian(prms::AbstractVector, locs::AbstractVector, dats::AbstractVector, opts::Maxlikopts,
                     d2funs::Vector{Vector{Function}}, strict::Bool=false)
  nllK = KernelMatrices.KernelMatrix(locs, locs, prms, opts.kernfun)
  tim1 = @elapsed begin
  HK   = KernelHODLR(nllK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  symmetricfactorize!(HK, plel=opts.fpll)
  end
  opts.verb && println("Assembly+factorize took      $(round(tim1, 3)) seconds.")
  tim2 = @elapsed begin
  Hess = stoch_hessian(nllK,HK,dats,opts.dfuns,d2funs,opts.saav,plel=opts.apll,
                       verbose=false,shuffle=!(opts.saa_fix))
  end
  opts.verb && println("Hessian took                 $(round(tim2, 3)) seconds.")
  opts.verb && println()
  return Hess
end

# Negative Log PROFILE likelihood Hessian
function nlpl_hessian(prms::AbstractVector, locs::AbstractVector, dats::AbstractVector, opts::Maxlikopts,
                      d2funs::Vector{Vector{Function}}, strict::Bool=false)
  nllK = KernelMatrices.KernelMatrix(locs, locs, prms, opts.kernfun)
  tim1 = @elapsed begin
  HK   = KernelHODLR(nllK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  symmetricfactorize!(HK, plel=opts.fpll)
  end
  opts.verb && println("Assembly+factorize took      $(round(tim1, 3)) seconds.")
  tim2 = @elapsed begin
  Hess = stoch_hessian(nllK,HK,dats,opts.dfuns,d2funs,opts.saav,plel=opts.apll,
                        verbose=false,shuffle=!(opts.saa_fix),profile=true)
  end
  opts.verb && println("Hessian took                 $(round(tim2, 3)) seconds.")
  opts.verb && println()
  return Hess
end

function fisher_matrix(prms::AbstractVector, locs::AbstractVector, dats::AbstractVector, opts::Maxlikopts)
  nllK = KernelMatrices.KernelMatrix(locs, locs, prms, opts.kernfun)
  HK   = KernelHODLR(nllK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  symmetricfactorize!(HK, plel=opts.fpll)
  DKs  = map(df -> DerivativeHODLR(nllK, df, HK, plel=opts.apll), opts.dfuns)
  fish = stoch_fisher(nllK, HK, DKs, opts.saav, plel=opts.apll, shuffle=!(opts.saa_fix))
  return fish
end

function gpsimulate(locs::AbstractVector, parms::Vector, opts::Maxlikopts; 
                    exact::Bool=false, kdtreesort::Bool=false)
  # Get the locations, sorted if requested:
  lcss = kdtreesort ? NearestNeighbors.KDTree(locs).data : locs
  # Allocate for the output:
  out  = zeros(length(lcss))
  inp  = randn(length(lcss))
  # Create the KernelMatrix, and either get its exact Cholesky or use HODLR to simulate it:
  covK = KernelMatrices.KernelMatrix(lcss, lcss, parms, opts.kernfun) 
  if exact
    cKf  = chol(Symmetric(full(covK)))
    mul!(out, transpose(cKf), inp)
  else
    cHK  = HODLR.KernelHODLR(covK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
    HODLR.symmetricfactorize!(cHK, plel=opts.fpll)
    mul!(out, cHK.W, inp)
  end
  return lcss, out
end

