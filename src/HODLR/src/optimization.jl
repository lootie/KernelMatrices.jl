
function _subproblem(xv::Vector, fx::Float64, gx::Vector, hx::Symmetric{Float64, Matrix{Float64}}, pv::Vector)
  return fx + dot(gx, pv) + 0.5*dot(pv,hx*pv)
end

function _rho(objfxp::Float64, x::Vector, fx::Float64, gx::Vector, 
              hx::Symmetric{Float64, Matrix{Float64}}, p::Vector)
  numr = fx - objfxp
  denm = _subproblem(x,fx,gx,hx,zeros(length(p))) - _subproblem(x,fx,gx,hx,p)
  return numr/denm
end

function _solve_subproblem_exact(g::Vector, B::Symmetric{Float64, Matrix{Float64}}, del::Float64)
  lmin = max(0.0, -real(eigmin(B)))
  lval = lmin + 0.1
  pl   = zeros(length(g))
  for cnt in 0:10
    Bi    = Symmetric(B + I*lval)
    R     = cholesky(Bi).U
    pl    = -Bi\g
    ql    = R\pl
    lval += abs2(norm(pl)/norm(ql))*(norm(pl)-del)/del
    (lval < lmin + 1.0e-10) && break
  end
  return pl
end

function trustregion(init::Vector, loc_s::AbstractVector, dat_s::AbstractVector,
                     d2funs::Vector{Vector{Function}}, opts::Maxlikopts; profile::Bool=false,
                     strict::Bool=false, vrb::Bool=false, dmax::Float64=1.0, dini::Float64=0.5,
                     eta::Float64=0.125, rtol::Float64=1.0e-8, atol::Float64=1.0e-5,
                     maxit::Int64=200, dcut::Float64=1.0e-4)
  dl, r1, st, cnt, fg = dini, 0.0, 0, 0, false
  xv = deepcopy(init)
  ro = zeros(length(init))
  gx = zeros(length(init))
  hx = zeros(length(init), length(init))
  for ct in 0:maxit
    cnt += 1
    vrb && println(xv)
    t1 = @elapsed begin
    if profile
      fx = nlpl_objective(xv, gx, loc_s, dat_s, opts)
    else
      fx = nll_objective(xv, gx, loc_s, dat_s, opts)
    end
    end
    vrb && println("objective+gradient  call took this long:  $(round(t1, digits=4))")
    t2 = @elapsed begin
    if profile
      hx .= nlpl_hessian(xv, loc_s, dat_s, opts, d2funs, strict)
    else
      hx .= nll_hessian(xv, loc_s, dat_s, opts, d2funs, strict)
    end
    end
    vrb && println("Hessian call took this long:              $(round(t2, digits=4))")
    vrb && println("Total time for the iteration:             $(round(t1+t2, digits=4))")
    vrb && println()
    # Solve the corresponding sub-problem:
    ro  .= _solve_subproblem_exact(gx, Symmetric(hx), dl)
    if profile
      fxp  = nlpl_objective(xv.+ro, Array{Float64}(undef, 0), loc_s, dat_s, opts)
    else
      fxp  = nll_objective(xv.+ro, Array{Float64}(undef, 0), loc_s, dat_s, opts)
    end
    r1   = _rho(fxp, xv, fx, gx, Symmetric(hx), ro)
    # Perform tests on solution of sub-problem:
    (r1 < 0.25)    && (dl = 0.25*norm(ro))
    (r1 > 0.75)    && (dl = min(2.0*norm(ro), dmax))
    (r1 > eta)     && (xv .+= ro)
    # Perform test for relative tolerance being reached:
    (fx != fxp && (isapprox(fx, fxp, rtol=rtol) || isapprox(fx, fxp, atol=atol))) && (fg = true)
    # Test for stopping the loop:
    fg             && (vrb && println("STOP: tolerance reached") ;          println() ; break)
    (norm(gx)<rtol)&& (vrb && println("STOP: Gradient reached tolerance") ; println() ; break)
    (dl<dcut)      && (vrb && println("STOP: Size of region too small")   ; println() ; break)
  end
  vrb && println("Total number of calls: $cnt")
  if profile
    pushfirst!(xv, nlpl_scale(xv, loc_s, dat_s, opts))
  end
  return xv
end

