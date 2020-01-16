
# Objective function struct that carries around the factorized kernel matrix
# If you call this struct with the arguments expected by Ipopt for
# eval_f, eval_grad_f, or eval_h, it will provide the corresponding output
mutable struct ObjectiveHODLR <: Function
  data::Maxlikdata
  opts::Maxlikopts
  K::KernelMatrix
  HK::KernelHODLR
  hess_type::Symbol  # :LBFGS, :FISHER, or :HESSIAN
end

function ObjectiveHODLR(init::Vector{Float64}, data::Maxlikdata,
                        opts::Maxlikopts, hess_type::Symbol)
  K  = KernelMatrix(data.pts_s, data.pts_s, init, opts.kernfun)
  HK = KernelHODLR(K, 0.0, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  symmetricfactorize!(HK, plel=opts.fpll)
  return ObjectiveHODLR(
    data, opts, K, HK, hess_type
  )
end

function update!(obj::ObjectiveHODLR, x::Vector{Float64})
  # Updata K and HK to reflect the new parameters
  obj.K.parms = NTuple{length(x)}(x)
  obj.HK = KernelHODLR(obj.K, 0.0, obj.opts.mrnk, obj.opts.lvl, nystrom=true, plel=obj.opts.apll)
  symmetricfactorize!(obj.HK, plel=obj.opts.fpll)
end

function (obj::ObjectiveHODLR)(x::Vector{Float64})
  # Update the kernel matrix if necessary
  sum(x .== obj.K.parms) != length(x) && update!(obj, x)
  # Compute the objective function
  return 0.5*logdet(obj.HK) + 0.5*dot(obj.data.dat_s, obj.HK\obj.data.dat_s)
end

function (obj::ObjectiveHODLR)(x::Vector{Float64}, target::Vector{Float64})
  # Update the kernel matrix if necessary
  sum(x .== obj.K.parms) != length(x) && update!(obj, x)
  # Compute the gradient
  tim1 = @elapsed begin
    target .= stoch_gradient(obj.K, obj.HK, obj.data.dat_s, obj.opts.dfuns, obj.opts.saav,
                             plel=obj.opts.apll, shuffle=!(obj.opts.saa_fix))
  end
  obj.opts.verb && println("Gradient took                 $(round(tim1, digits=3)) seconds.")
  obj.opts.verb && println()
end

function (obj::ObjectiveHODLR)(x::Vector{Float64}, mode::Symbol,
                               rows::Vector{Int32}, cols::Vector{Int32},
                               obj_factor::Float64, lambda::Vector{Float64},
                               target::Vector{Float64})
  if mode == :Structure
    # Provide the nonzero structure of the Hessian
    # This is just all pairs of indices in the lower triangle
    row_inds, col_inds = map(k->getfield.([(i,j) for i=1:length(obj.K.parms) for j=1:i], k), 1:2)
    rows .= row_inds
    cols .= col_inds
  elseif mode == :Values
    # Update the kernel matrix if necessary
    sum(x .== obj.K.parms) != length(x) && update!(obj, x)
    # Compute the Hessian or the Fisher matrix
    tim1 = @elapsed begin
      if obj.hess_type == :HESSIAN
        Hess = stoch_hessian(obj.K, obj.HK, obj.data.dat_s, obj.opts.dfuns, obj.opts.d2funs, obj.opts.saav,
                             plel=obj.opts.apll, shuffle=!(obj.opts.saa_fix))
      elseif obj.hess_type == :FISHER
        DKs  = map(df -> DerivativeHODLR(obj.K, df, obj.HK, plel=obj.opts.apll), obj.opts.dfuns)
        Hess = stoch_fisher(obj.K, obj.HK, DKs, obj.opts.saav,
                            plel=obj.opts.apll, shuffle=!(obj.opts.saa_fix))
      end
      target .= obj_factor * [Hess[i,j] for i=1:length(x) for j=1:i]
    end
    obj.opts.verb && println("Hessian took                 $(round(tim1, digits=3)) seconds.")
    obj.opts.verb && println()
  end
end

function Maxlikproblem(init::AbstractVector, data::Maxlikdata, opts::Maxlikopts;
                       lb::AbstractVector=fill(-1e20,length(init)),
                       ub::AbstractVector=fill(1e20,length(init)),
                       hess_type::Symbol=:LBFGS)
  hess_type in [:LBFGS, :FISHER, :HESSIAN] || error("Please select hess_type :LBFGS, :FISHER, or :HESSIAN")
  obj  = ObjectiveHODLR(init, data, opts, hess_type)
  n    = length(init)
  prob = createProblem(
    n,                     # Number of variables
    lb,                    # Objective lower bounds (setting this below -1e19 means no lower bound)
    ub,                    # Objective upper bounds (setting this above  1e19 means no upper bound)
    0,                     # Number of constraints
    Vector{Float64}([]),   # Constraint lower bounds
    Vector{Float64}([]),   # Constraint upper bounds
    0,                     # Number of nonzero elements in the Jacobian of the constraints
    Int64(n*(n+1)/2),      # Number of nonzero elements in the Hessian of the Lagrangian
    obj,                                # Objective
    (args...) -> nothing,               # Constraints
    obj,                                # Gradient of objective
    (args...) -> nothing,               # Jacobian of constraints
    hess_type == :LBFGS ? nothing : obj # Hessian of Lagrangian
  )
  hess_type == :LBFGS && addOption(prob, "hessian_approximation", "limited-memory")

  return prob
end
