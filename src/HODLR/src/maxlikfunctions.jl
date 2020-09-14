
function negloglik(HK::Union{Matrix{T}, KernelHODLR{T}}, dat::Vector{T}) where{T<:Number}
  typeof(HK) == KernelHODLR{T} || warn("This may take a while on a full matrix...")
  if typeof(HK) == KernelHODLR{T} && HK.U != nothing
    error("The HODLR matrix needs to be factorized for this.")
  end
  nll = 0.5*logdet(HK) + 0.5*dot(dat, HK\dat)
  return nll
end

function profile_negloglik(HK::Union{Matrix{T}, KernelHODLR{T}}, 
                           dat::Vector{T}) where{T<:Number}
  typeof(HK) == KernelHODLR{T} || warn("This may take a while on a full matrix...")
  if typeof(HK) == KernelHODLR{T} && HK.U != nothing
    error("The HODLR matrix needs to be factorized for this.")
  end
  nll = logdet(HK) + length(dat)*log(dot(dat, HK\dat))
  return 0.5*nll
end

function scaleparm_mle(prms::AbstractVector, data::Maxlikdata, opts::Maxlikopts)
  nllK = KernelMatrix(data.pts_s, data.pts_s, prms, opts.kernfun)
  HK   = KernelHODLR(nllK, 0.0, opts.mrnk, opts.lvl, nystrom=true, 
                     plel=opts.apll, sorted=true)
  symmetricfactorize!(HK, plel=opts.fpll)
  return dot(data.dat_s, HK\data.dat_s)/length(data.dat_s)
end

function nll_objective(prms::AbstractVector, grad::Vector, 
                       data::Maxlikdata, opts::Maxlikopts)
  opts.verb && @show prms
  nllK = KernelMatrices.KernelMatrix(data.pts_s, data.pts_s, prms, opts.kernfun)
  tim1 = @elapsed begin
  HK   = KernelHODLR(nllK, 0.0, opts.mrnk, opts.lvl, nystrom=true, 
                     plel=opts.apll, sorted=true)
  symmetricfactorize!(HK, plel=opts.fpll)
  end
  opts.verb && println("Assembly+factorize took       $(round(tim1, digits=3)) seconds.")
  tim2 = @elapsed nll     = negloglik(HK, data.dat_s)
  opts.verb && println("Negative log-lik took         $(round(tim2, digits=3)) seconds.")
  tim3 = @elapsed begin
  if length(grad) > 0
    grad .= stoch_gradient(nllK, HK, data.dat_s, opts.dfuns, opts.saav, 
                           plel=opts.apll, shuffle=!(opts.saa_fix))
  end
  end
  opts.verb && println("Gradient took                 $(round(tim3, digits=3)) seconds.")
  opts.verb && println()
  return nll
end

function nlpl_scale(prms::AbstractVector, data::Maxlikdata, opts::Maxlikopts)
  nllK = KernelMatrices.KernelMatrix(data.pts_s, data.pts_s, prms, opts.kernfun)
  HK   = KernelHODLR(nllK, 0.0, opts.mrnk, opts.lvl, nystrom=true, 
                     plel=opts.apll, sorted=true)
  symmetricfactorize!(HK, plel=opts.fpll)
  return dot(data.dat_s, HK\data.dat_s)/length(data.dat_s)
end

# Negative Log PROFILE likelihood
function nlpl_objective(prms::AbstractVector, grad::Vector, 
                        data::Maxlikdata, opts::Maxlikopts)
  opts.verb && @show prms
  nllK = KernelMatrices.KernelMatrix(data.pts_s, data.pts_s, prms, opts.kernfun)
  tim1 = @elapsed begin
  HK   = KernelHODLR(nllK, 0.0, opts.mrnk, opts.lvl, nystrom=true, 
                     plel=opts.apll, sorted=true)
  symmetricfactorize!(HK, plel=opts.fpll)
  end
  opts.verb && println("Assembly+factorize took       $(round(tim1, digits=3)) seconds.")
  tim2 = @elapsed nll     = profile_negloglik(HK, data.dat_s)
  opts.verb && println("Negative log-lik took         $(round(tim2, digits=3)) seconds.")
  tim3 = @elapsed begin
  if length(grad) > 0
    grad .= stoch_profile_gradient(nllK, HK, data.dat_s, opts.dfuns, opts.saav, 
                                   plel=opts.apll, shuffle=!(opts.saa_fix))
  end
  end
  opts.verb && println("Gradient took                 $(round(tim3, digits=3)) seconds.")
  opts.verb && println()
  return nll
end

function nll_gradient(prms::AbstractVector, data::Maxlikdata, opts::Maxlikopts)
  nllK = KernelMatrices.KernelMatrix(data.pts_s, data.pts_s, prms, opts.kernfun)
  tim1 = @elapsed begin
  HK   = KernelHODLR(nllK, 0.0, opts.mrnk, opts.lvl, nystrom=true, 
                     plel=opts.apll, sorted=true)
  symmetricfactorize!(HK, plel=opts.fpll)
  end
  opts.verb && println("Assembly+factorize took       $(round(tim1, digits=3)) seconds.")
  tim2 = @elapsed begin
    grad = stoch_gradient(nllK, HK, data.dat_s, opts.dfuns, opts.saav, 
                          plel=opts.apll, shuffle=!(opts.saa_fix))
  end
  opts.verb && println("Gradient took                 $(round(tim2, digits=3)) seconds.")
  opts.verb && println()
  return grad
end

function nll_hessian(prms::AbstractVector, data::Maxlikdata, opts::Maxlikopts)
  isnothing(opts.d2funs) && throw(error("You need to supply second derivatives for this."))
  nllK = KernelMatrices.KernelMatrix(data.pts_s, data.pts_s, prms, opts.kernfun)
  tim1 = @elapsed begin
  HK   = KernelHODLR(nllK, 0.0, opts.mrnk, opts.lvl, nystrom=true, 
                     plel=opts.apll, sorted=true)
  symmetricfactorize!(HK, plel=opts.fpll)
  end
  opts.verb && println("Assembly+factorize took      $(round(tim1, digits=3)) seconds.")
  tim2 = @elapsed begin
  Hess = stoch_hessian(nllK,HK,data.dat_s,opts.dfuns,opts.d2funs,opts.saav,plel=opts.apll,
                       verbose=false,shuffle=!(opts.saa_fix))
  end
  opts.verb && println("Hessian took                 $(round(tim2, digits=3)) seconds.")
  opts.verb && println()
  return Hess
end

# Negative Log PROFILE likelihood Hessian
function nlpl_hessian(prms::AbstractVector, data::Maxlikdata, opts::Maxlikopts)
  isnothing(opts.d2funs) && throw(error("You need to supply second derivatives for this."))
  nllK = KernelMatrices.KernelMatrix(data.pts_s, data.pts_s, prms, opts.kernfun)
  tim1 = @elapsed begin
  HK   = KernelHODLR(nllK, 0.0, opts.mrnk, opts.lvl, nystrom=true, 
                     plel=opts.apll, sorted=true)
  symmetricfactorize!(HK, plel=opts.fpll)
  end
  opts.verb && println("Assembly+factorize took      $(round(tim1, digits=3)) seconds.")
  tim2 = @elapsed begin
  Hess = stoch_hessian(nllK,HK,data.dat_s,opts.dfuns,opts.d2funs,opts.saav,plel=opts.apll,
                        verbose=false,shuffle=!(opts.saa_fix),profile=true)
  end
  opts.verb && println("Hessian took                 $(round(tim2, digits=3)) seconds.")
  opts.verb && println()
  return Hess
end

function fisher_matrix(prms::AbstractVector, data::Maxlikdata, opts::Maxlikopts)
  nllK = KernelMatrices.KernelMatrix(data.pts_s, data.pts_s, prms, opts.kernfun)
  HK   = KernelHODLR(nllK, 0.0, opts.mrnk, opts.lvl, nystrom=true, 
                     plel=opts.apll, sorted=true)
  symmetricfactorize!(HK, plel=opts.fpll)
  DKs  = map(df -> DerivativeHODLR(nllK, df, HK, plel=opts.apll), opts.dfuns)
  fish = stoch_fisher(nllK, HK, DKs, opts.saav, plel=opts.apll, shuffle=!(opts.saa_fix))
  return fish
end

function gpsimulate(kernel::Function, locs::AbstractVector, parms::Vector, 
                    opts::Union{Nothing, Maxlikopts}=nothing; 
                    exact::Bool=false, kdtreesort::Bool=false)
  if !kdtreesort 
  @warn "Not sorting data can adversely affect approximation quality..." maxlog=1
  end
  # Get the locations, sorted if requested:
  lcss = kdtreesort ? NearestNeighbors.KDTree(locs).data : locs
  # Allocate for the output:
  out  = zeros(length(lcss))
  inp  = randn(length(lcss))
  # Create the KernelMatrix, and either get its exact Cholesky or use HODLR to simulate it:
  covK = KernelMatrix(lcss, lcss, parms, kernel) 
  if exact
    cKf  = cholesky(Symmetric(full(covK))).U
    mul!(out, transpose(cKf), inp)
  else
    opts == nothing && throw(error("Supply HODLR options for approximated simulations."))
    cHK  = HODLR.KernelHODLR(covK, 0.0, opts.mrnk, opts.lvl, 
                             nystrom=true, plel=opts.apll)
    HODLR.symmetricfactorize!(cHK, plel=opts.fpll)
    mul!(out, cHK.W, inp)
  end
  return maxlikdata(pts=lcss, data=out, sortmethod=:Nothing, warn=false)
end

function gpsimulate(kernel::Function, parms::Vector, n::Int64, dim::Int64,
                    boxsz::Float64, opts::Union{Nothing, Maxlikopts}=nothing;
                    exact::Bool=false)
  pts = [SVector{dim}(rand(dim).*boxsz) for _ in 1:n]
  return gpsimulate(kernel, pts, parms, opts, exact=exact, kdtreesort=true)
end

