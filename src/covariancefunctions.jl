
##
#
# Inverse quadratic:
#
##

function rat_kernfun(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector, numnug::Bool=true)
  out = parms[1]/abs2(1.0 + abs2(norm(x1-x2)/parms[2]))
  if numnug && x1 == x2   # A numerical nugget. This will make numerics easier
    out += 1.0e-12        # because this function is analytic everywhere,
  end                     # including at the origin, which makes numerics annoying.
  return out
end

function rat_kernfun_d1(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  out = 1.0/abs2(1.0 + abs2(norm(x1-x2)/parms[2]))
  return out
end

function rat_kernfun_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nx12 = norm(x1-x2)
  alp  = 4.0*parms[1]*abs2(nx12)
  out  = alp*(parms[2] + abs2(nx12)/parms[2])^(-3)
  return out
end

function rat_kernfun_d1_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nx12 = norm(x1-x2)
  alp  = 4.0*abs2(nx12)
  out  = alp*(parms[2] + abs2(nx12)/parms[2])^(-3)
  return out
end

function rat_kernfun_d2_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nx12 = norm(x1-x2)
  alp  = 4.0*parms[1]*abs2(nx12)
  out  = -3.0*alp*(1.0 + -abs2(nx12/parms[2]))/abs2(abs2(parms[2] + abs2(nx12)/parms[2]))
  return out
end

##
#
# Profiled inverse quadratic:
#
##

function prt_kernfun(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector, numnug::Bool=true)
  out = 1.0/abs2(1.0 + abs2(norm(x1-x2)/parms[1]))
  if numnug && x1 == x2  # same deal about numerical nugget. See rat_kernfun.
    out += 1.0e-12
  end
  return out
end

function prt_kernfun_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nx12 = norm(x1-x2)
  alp  = 4.0*abs2(nx12)
  out  = alp*(parms[1] + abs2(nx12)/parms[1])^(-3)
  return out
end

function prt_kernfun_d2_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nx12 = norm(x1-x2)
  alp  = 4.0*abs2(nx12)
  out  = -3.0*alp*(1.0 + -abs2(nx12/parms[1]))/abs2(abs2(parms[1] + abs2(nx12)/parms[1]))
  return out
end


##
#
# Matern: (using parameterization of Stein 2013):
#
##

function give_eps(x::Number)
  cbrt(eps(Float64)) * max(1.0, abs(x)) 
end

function gamma_dx(x::Number)
  (x < 0.0 && isinteger(x)) && error("domain problem")
  digamma(x)*gamma(x)
end

function gamma_dx_dx(x::Number)
  (x < 0.0 && isinteger(x)) && error("domain problem")
  gamma(x)*(polygamma(1, x) + abs2(digamma(x)))
end

function besselk_dx(nu::Number, x::Number)
  -0.5*(besselk(nu+1.0, x) + besselk(nu-1.0, x))
end

function besseli_dnu(nu::Float64, x::Float64; maxit::Int64 = 1_000_000, 
                     rtol::Float64 = 1.0e-12)::Float64
  (isinteger(nu) && isless(nu, 0)) && error(DomainError(nu, "Don't call with negative ints."))
  sum = 0.0
  for j in 0:maxit
    new   = digamma(j+nu+1)*(0.25*abs2(x))^j/(gamma(j+nu+1)*gamma(j+1))
    sum  += new
    abs(new/sum) < rtol && break
  end
  return besseli(nu, x)*log(0.5*x) - sum*(0.5*x)^nu
end

function besselk_dnu(nu::Float64, x::Float64)
  rnu = round(nu, digits=5)
  isinteger(rnu)   && return besselk_dnu(Int64(rnu), x)
  isless(100.0, x) && return 0.0
  tmp1 = -besseli_dnu(-nu, x)                  # If the besseli has exploded, use finite diff,
  (abs(tmp1) > 1.0e5 || isnan(tmp1)) && begin  # as subtracting two numbers that have order of 
    ep = give_eps(nu)                          # magnitude 10^18 or whatever will cause failures.
    return (besselk(nu+ep, x)-besselk(nu-ep, x))/(2.0*ep)
  end
  bdff = tmp1 - besseli_dnu(nu, x)
  return 0.5*pi*csc(nu*pi)*bdff - pi*cot(nu*pi)*besselk(nu, x)
end

function besselk_dnu(nu::Int64, x::Float64)
  sm = 0.0
  for j in 0:(nu-1)
    sm += besselk(j, x)*(0.5*x)^j/(factorial(j)*(nu-j))
  end
  return factorial(nu)*sm/(2.0*(0.5*x)^nu)
end

function besselk_dx_dx(nu::Number, x::Number)
  -0.5*(besselk_dx(nu+1.0, x) + besselk_dx(nu-1.0, x))
end

function besselk_dx_dnu(nu::Number, x::Number)
  -0.5*(besselk_dnu(nu+1.0, x) + besselk_dnu(nu-1.0, x))
end

function besselk_dnu_dnu(nu::Number, x::Number)
  ep = give_eps(nu)
  (besselk_dnu(nu+ep, x)-besselk_dnu(nu-ep, x))/(2.0*ep)
end

function mtn_p1(nu::Number, x::Number)
  (sqrt(2.0*nu)*x)^nu
end

function mtn_p1_dx(nu::Number, x::Number)
  sqrt(2.0*nu)*nu*(sqrt(2.0*nu)*x)^(nu-1.0)
end

function mtn_p1_dnu(nu::Number, x::Number)
  ((sqrt(2.0*nu)*x)^nu)*(log(sqrt(2.0*nu)*x)+0.5)
end

function mtn_p1_dx_dx(nu::Number, x::Number)
  2.0*abs2(nu)*(nu-1.0)*(sqrt(2.0*nu)*x)^(nu-2.0)
end

function mtn_p1_dnu_dx(nu::Number, x::Number)
  mtn_p1_dx(nu, x)*(log(sqrt(2.0*nu)*x) + 0.5) + mtn_p1(nu, x)/x
end

function mtn_p1_dnu_dnu(nu::Number, x::Number)
  mtn_p1_dnu(nu, x)*(log(sqrt(2.0*nu)*x) + 0.5) + mtn_p1(nu, x)/(2.0*nu)
end

function mtn_p2(nu::Number, x::Number)
  besselk(nu, sqrt(2.0*nu)*x)
end

function mtn_p2_dx(nu::Number, x::Number)
  besselk_dx(nu, sqrt(2.0*nu)*x)*sqrt(2.0*nu)
end

function mtn_p2_dnu(nu::Number, x::Number)
  besselk_dnu(nu, sqrt(2.0*nu)*x) + x*besselk_dx(nu, sqrt(2.0*nu)*x)/sqrt(2.0*nu)
end

function mtn_p2_dx_dx(nu::Number, x::Number)
  besselk_dx_dx(nu, sqrt(2.0*nu)*x)*(2.0*nu)
end

function mtn_p2_dnu_dx(nu::Number, x::Number)
  out  = besselk_dx_dnu(nu, sqrt(2.0*nu)*x)*sqrt(2.0*nu)
  out += (besselk_dx(nu, sqrt(2.0*nu)*x)/sqrt(2.0*nu) + x*besselk_dx_dx(nu, sqrt(2.0*nu)*x))
  return out
end

# Central finite difference, because my own home-rolled bessel expansion wasn't actually more precise.
function mtn_p2_dnu_dnu(nu::Number, x::Number)
  ep = give_eps(nu)
  (mtn_p2_dnu(nu+ep, x)-mtn_p2_dnu(nu-ep, x))/(2.0*ep)
end

function mtn_p3(nu::Number)
  1.0/(gamma(nu)*2.0^(nu-1.0))
end

function mtn_p3_dnu(nu::Number)
  out = gamma_dx(nu)*(2.0^(nu-1.0)) + gamma(nu)*(2.0^(nu-1.0))*log(2.0)
  out/= abs2(gamma(nu)*(2.0^(nu-1.0)))
  return -out
end

# Central finite difference, because my own home-rolled bessel expansion wasn't actually more precise.
function mtn_p3_dnu_dnu(nu::Number)
  ep = give_eps(nu)
  (mtn_p3_dnu(nu+ep)-mtn_p3_dnu(nu-ep))/(2.0*ep)
end

function mtn_cor(nu::Number, x::Number)
  isapprox(x, zero(x), atol=1.0e-8) && return 1.0
  mtn_p1(nu,x)*mtn_p2(nu,x)*mtn_p3(nu)
end

function mtn_cor_dx(nu::Number, x::Number)
  isapprox(x, zero(x), atol=1.0e-8) && return 0.0
  mtn_p3(nu)*(mtn_p1_dx(nu,x)*mtn_p2(nu,x) + mtn_p1(nu,x)*mtn_p2_dx(nu,x))
end

function mtn_cor_dnu(nu::Number, x::Number)
  isapprox(x, 0.0, atol=1.0e-8) && return 0.0
  out  = mtn_p1_dnu(nu, x)*mtn_p2(nu, x)*mtn_p3(nu)
  out += mtn_p1(nu, x)*mtn_p2_dnu(nu, x)*mtn_p3(nu)
  out += mtn_p1(nu, x)*mtn_p2(nu, x)*mtn_p3_dnu(nu)
  return out
end

function mtn_cor_dx_dx(nu::Number, x::Number)
  ird  = mtn_p1_dx_dx(nu,x)*mtn_p2(nu,x) + mtn_p1_dx(nu,x)*mtn_p2_dx(nu,x)
  ird += mtn_p1_dx(nu,x)*mtn_p2_dx(nu,x) + mtn_p1(nu,x)*mtn_p2_dx_dx(nu,x)
  out  = mtn_p3(nu)*ird
  return out
end

function mtn_cor_dnu_dx(nu::Number, x::Number)
  out  = mtn_p3_dnu(nu)*(mtn_p1_dx(nu,x)*mtn_p2(nu,x) + mtn_p1(nu,x)*mtn_p2_dx(nu,x))
  ird  = mtn_p1_dnu_dx(nu,x)*mtn_p2(nu,x) + mtn_p1_dx(nu,x)*mtn_p2_dnu(nu,x)
  ird += mtn_p1_dnu(nu,x)*mtn_p2_dx(nu,x) + mtn_p1(nu,x)*mtn_p2_dnu_dx(nu,x)
  out += mtn_p3(nu)*ird
  return out
end

function mtn_cor_dnu_dnu(nu::Number, x::Number)
  out  = mtn_p1_dnu_dnu(nu,x)*mtn_p2(nu,x)*mtn_p3(nu)
  out += mtn_p1_dnu(nu,x)*mtn_p2_dnu(nu,x)*mtn_p3(nu)
  out += mtn_p1_dnu(nu,x)*mtn_p2(nu,x)*mtn_p3_dnu(nu)
  out += mtn_p1_dnu(nu,x)*mtn_p2_dnu(nu,x)*mtn_p3(nu)
  out += mtn_p1(nu,x)*mtn_p2_dnu_dnu(nu,x)*mtn_p3(nu)
  out += mtn_p1(nu,x)*mtn_p2_dnu(nu,x)*mtn_p3_dnu(nu)
  out += mtn_p1_dnu(nu,x)*mtn_p2(nu,x)*mtn_p3_dnu(nu)
  out += mtn_p1(nu,x)*mtn_p2_dnu(nu,x)*mtn_p3_dnu(nu)
  out += mtn_p1(nu,x)*mtn_p2(nu,x)*mtn_p3_dnu_dnu(nu)
  return out
end

function mtn_kernfun(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], parms[3]
  nx12       = norm(x1-x2)
  out        = t0
  if nx12 > 0.0
    out *= mtn_cor(nu, nx12/t1)
  end
  return out
end

function mtn_kernfun_d1(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], parms[3]
  out  = 1.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out = mtn_cor(nu, nx12/t1)
  end
  return out
end

function mtn_kernfun_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], parms[3]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = -t0*mtn_cor_dx(nu, nx12/t1)*nx12/abs2(t1)
  end
  return out
end

function mtn_kernfun_d3(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], parms[3]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = t0*mtn_cor_dnu(nu, nx12/t1)
  end
  return out
end

function mtn_kernfun_d1_d1(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  return 0.0 ;
end

function mtn_kernfun_d1_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], parms[3]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = -mtn_cor_dx(nu, nx12/t1)*nx12/abs2(t1)
  end
  return out
end

function mtn_kernfun_d1_d3(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], parms[3]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = mtn_cor_dnu(nu, nx12/t1)
  end
  return out
end

function mtn_kernfun_d2_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], parms[3]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out = t0*(mtn_cor_dx_dx(nu, nx12/t1)*abs2(nx12/abs2(t1)) + mtn_cor_dx(nu, nx12/t1)*2.0*nx12/(t1^3))
  end
  return out
end

function mtn_kernfun_d2_d3(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], parms[3]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = -t0*mtn_cor_dnu_dx(nu, nx12/t1)*nx12/abs2(t1)
  end
  return out
end

function mtn_kernfun_d3_d3(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], parms[3]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = t0*mtn_cor_dnu_dnu(nu, nx12/t1)
  end
  return out
end

##
#
# Profiled Matern: (using parameterization of Stein 2013):
#
##

function pmt_kernfun(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = 1.0, parms[1], parms[2]
  nx12       = norm(x1-x2)
  out        = t0
  if nx12 > 0.0
    out *= mtn_cor(nu, nx12/t1)
  end
  return out
end

function pmt_kernfun_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = 1.0, parms[1], parms[2]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = -t0*mtn_cor_dx(nu, nx12/t1)*nx12/abs2(t1)
  end
  return out
end

function pmt_kernfun_d3(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = 1.0, parms[1], parms[2]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = t0*mtn_cor_dnu(nu, nx12/t1)
  end
  return out
end

function pmt_kernfun_d2_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = 1.0, parms[1], parms[2]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out = t0*(mtn_cor_dx_dx(nu, nx12/t1)*abs2(nx12/abs2(t1)) + mtn_cor_dx(nu, nx12/t1)*2.0*nx12/(t1^3))
  end
  return out
end

function pmt_kernfun_d2_d3(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = 1.0, parms[1], parms[2]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = -t0*mtn_cor_dnu_dx(nu, nx12/t1)*nx12/abs2(t1)
  end
  return out
end

function pmt_kernfun_d3_d3(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = 1.0, parms[1], parms[2]
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = t0*mtn_cor_dnu_dnu(nu, nx12/t1)
  end
  return out
end

##
#
# Matern with fixed nu=1 (Stein 2013):
#
##

function mt1_kernfun(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], 1.0
  nx12       = norm(x1-x2)
  out        = t0
  if nx12 > 0.0
    out *= mtn_cor(nu, nx12/t1)
  end
  return out
end

function mt1_kernfun_d1(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], 1.0
  out  = 1.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out = mtn_cor(nu, nx12/t1)
  end
  return out
end

function mt1_kernfun_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], 1.0
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = -t0*mtn_cor_dx(nu, nx12/t1)*nx12/abs2(t1)
  end
  return out
end

function mt1_kernfun_d1_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], 1.0
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = -mtn_cor_dx(nu, nx12/t1)*nx12/abs2(t1)
  end
  return out
end

function mt1_kernfun_d2_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = parms[1], parms[2], 1.0
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out = t0*(mtn_cor_dx_dx(nu, nx12/t1)*abs2(nx12/abs2(t1)) + mtn_cor_dx(nu, nx12/t1)*2.0*nx12/(t1^3))
  end
  return out
end

##
#
# Profiled Matern with fixed nu=1 (Stein 2013):
#
##

function pm1_kernfun(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = 1.0, parms[1], 1.0
  nx12       = norm(x1-x2)
  out        = t0
  if nx12 > 0.0
    out *= mtn_cor(nu, nx12/t1)
  end
  return out
end

function pm1_kernfun_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = 1.0, parms[1], 1.0
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out  = -t0*mtn_cor_dx(nu, nx12/t1)*nx12/abs2(t1)
  end
  return out
end

function pm1_kernfun_d2_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1, nu = 1.0, parms[1], 1.0
  out = 0.0
  nx12 = norm(x1-x2)
  if nx12 > 0.0
    out = t0*(mtn_cor_dx_dx(nu, nx12/t1)*abs2(nx12/abs2(t1)) + mtn_cor_dx(nu, nx12/t1)*2.0*nx12/(t1^3))
  end
  return out
end

##
#
# Handcock Matern with fixed nu:
#
##

function hmt_p1()
  nu = 1.2
  return 1.0/(gamma(nu)*2.0^(nu-1.0))
end

function hmt_p2(t1::Number, x::Number)
  nu = 1.2
  (2.0*sqrt(nu)*x/t1)^nu
end

function hmt_p2_dt1(t1::Number, x::Number)
  nu = 1.2
  a1 = nu*(2.0*sqrt(nu)*x/t1)^(nu-1.0)
  a2 = -2.0*sqrt(nu)*x/abs2(t1)
  return a1*a2
end

function hmt_p2_dt1_dt1(t1::Number, x::Number)
  nu      = 1.2
  a1      = nu*(2.0*sqrt(nu)*x/t1)^(nu-1.0)
  a2      = -2.0*sqrt(nu)*x/abs2(t1)
  da1_dt1 = (nu*(nu-1.0)*(2.0*sqrt(nu)*x/t1)^(nu-2.0))*a2
  da2_dt1 = 4.0*sqrt(nu)*x/(t1^3)
  return a1*da2_dt1 + a2*da1_dt1
end

function hmt_p3(t1::Number, x::Number)
  nu = 1.2
  return besselk(nu, 2.0*sqrt(nu)*x/t1)
end

function hmt_p3_dt1(t1::Number, x::Number)
  nu = 1.2
  return besselk_dx(nu, 2.0*sqrt(nu)*x/t1) * (-2.0*sqrt(nu)*x/abs2(t1))
end

function hmt_p3_dt1_dt1(t1::Number, x::Number)
  nu      = 1.2
  a1      = besselk_dx(nu, 2.0*sqrt(nu)*x/t1)
  da1_dt1 = besselk_dx_dx(nu, 2.0*sqrt(nu)*x/t1) * (-2.0*sqrt(nu)*x/abs2(t1))
  a2      = (-2.0*sqrt(nu)*x/abs2(t1))
  da2_dt1 = 4.0*sqrt(nu)*x/(t1^3) 
  return a1*da2_dt1 + a2*da1_dt1
end

function hmt_cor(t1::Number, x::Number)
  hmt_p1()*hmt_p2(t1,x)*hmt_p3(t1,x)
end

function hmt_cor_dt1(t1::Number, x::Number)
  out  = hmt_p2_dt1(t1,x)*hmt_p3(t1,x)
  out += hmt_p2(t1,x)*hmt_p3_dt1(t1,x)
  return hmt_p1()*out
end

function hmt_cor_dt1_dt1(t1::Number, x::Number)
  out  = hmt_p2_dt1_dt1(t1,x)*hmt_p3(t1,x)
  out += hmt_p2_dt1(t1,x)*hmt_p3_dt1(t1,x)
  out += hmt_p2_dt1(t1,x)*hmt_p3_dt1(t1,x)
  out += hmt_p2(t1,x)*hmt_p3_dt1_dt1(t1,x)
  return hmt_p1()*out
end

function hmt_kernfun(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1 = parms[1], parms[2]
  out    = t0
  nx12   = norm(x1-x2)
  if nx12 > 0
    out *= hmt_cor(t1, nx12)
  end
  return out
end

function hmt_kernfun_d1(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1 = 1.0, parms[2]
  out    = t0
  nx12   = norm(x1-x2)
  if nx12 > 0
    out = hmt_cor(t1, nx12)
  end
  return out
end

function hmt_kernfun_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1 = parms[1], parms[2]
  out    = 0.0
  nx12   = norm(x1-x2)
  if nx12 > 0
    out = hmt_cor_dt1(t1, nx12)
  end
  return out
end

function hmt_kernfun_d1_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1 = 1.0, parms[2]
  out    = 0.0
  nx12   = norm(x1-x2)
  if nx12 > 0
    out = hmt_cor_dt1(t1, nx12)
  end
  return out
end

function hmt_kernfun_d2_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1 = parms[1], parms[2]
  out    = 0.0
  nx12   = norm(x1-x2)
  if nx12 > 0
    out = hmt_cor_dt1_dt1(t1, nx12)
  end
  return out
end

##
#
# Profiled Handcock Matern with fixed nu:
#
##

function pht_kernfun(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1 = 1.0, parms[1]
  out    = t0
  nx12   = norm(x1-x2)
  if nx12 > 0
    out *= hmt_cor(t1, nx12)
  end
  return out
end

function pht_kernfun_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1 = 1.0, parms[1]
  out    = 0.0
  nx12   = norm(x1-x2)
  if nx12 > 0
    out = hmt_cor_dt1(t1, nx12)
  end
  return out
end

function pht_kernfun_d2_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  t0, t1 = 1.0, parms[1]
  out    = 0.0
  nx12   = norm(x1-x2)
  if nx12 > 0
    out = hmt_cor_dt1_dt1(t1, nx12)
  end
  return out
end

##
#
# New Stein matern with fixed nu:
#
##

function sm1_kernfun(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nu    = 1.0
  t0,t1 = parms[1], parms[2]
  nx12  = norm(x1-x2)
  out   = t0*(abs2(t1)/(4.0*nu))*(gamma(nu)*(2.0^(nu-1.0)))
  if nx12 != 0.0
    a1  = (nx12*t1/(2.0*sqrt(nu)))^nu
    a2  = besselk(nu, 2.0*sqrt(nu)*nx12/t1)
    out = t0*a1*a2
  end
  return out
end

function sm1_kernfun_d1(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nu    = 1.0
  t0,t1 = parms[1], parms[2]
  nx12  = norm(x1-x2)
  out   = (abs2(t1)/(4.0*nu))*(gamma(nu)*(2.0^(nu-1.0)))
  if nx12 != 0.0
    a1  = (nx12*t1/(2.0*sqrt(nu)))^nu
    a2  = besselk(nu, 2.0*sqrt(nu)*nx12/t1)
    out = a1*a2
  end
  return out
end

function sm1_kernfun_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nu    = 1.0
  t0,t1 = parms[1], parms[2]
  nx12  = norm(x1-x2)
  out   = t0*2.0*(t1/(4.0*nu))*(gamma(nu)*(2.0^(nu-1.0)))
  if nx12 != 0.0
    a1    = (nx12*t1/(2.0*sqrt(nu)))^nu
    a1_d2 = (nu*(nx12*t1/(2.0*sqrt(nu)))^(nu-1.0))*(nx12/2.0*sqrt(nu))
    a2    = besselk(nu, 2.0*sqrt(nu)*nx12/t1)
    a2_d2 = -besselk_dx(nu, 2.0*sqrt(nu)*nx12/t1)*2.0*sqrt(nu)*nx12/abs2(t1) 
    out = t0*(a1*a2_d2 + a1_d2*a2)
  end
  return out
end

function sm1_kernfun_d1_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nu    = 1.0
  t0,t1 = parms[1], parms[2]
  nx12  = norm(x1-x2)
  out   = 2.0*(t1/(4.0*nu))*(gamma(nu)*(2.0^(nu-1.0)))
  if nx12 != 0.0
    a1    = (nx12*t1/(2.0*sqrt(nu)))^nu
    a1_d2 = (nu*(nx12*t1/(2.0*sqrt(nu)))^(nu-1.0))*(nx12/2.0*sqrt(nu))
    a2    = besselk(nu, 2.0*sqrt(nu)*nx12/t1)
    a2_d2 = -besselk_dx(nu, 2.0*sqrt(nu)*nx12/t1)*2.0*sqrt(nu)*nx12/abs2(t1) 
    out = (a1*a2_d2 + a1_d2*a2)
  end
  return out
end

function sm1_kernfun_d2_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nu    = 1.0
  t0,t1 = parms[1], parms[2]
  nx12  = norm(x1-x2)
  out   = t0*(2.0/(4.0*nu))*(gamma(nu)*(2.0^(nu-1.0)))
  if nx12 != 0.0
    a1       = (nx12*t1/(2.0*sqrt(nu)))^nu
    a1_d2    = (nu*(nx12*t1/(2.0*sqrt(nu)))^(nu-1.0))*(nx12/2.0*sqrt(nu))
    a1_d2_d2 = (nu*(nu-1.0)*(nx12*t1/(2.0*sqrt(nu)))^(nu-2.0))*abs2(nx12/(2.0*sqrt(nu)))
    a2       = besselk(nu, 2.0*sqrt(nu)*nx12/t1)
    a2_d2    = -besselk_dx(nu, 2.0*sqrt(nu)*nx12/t1)*2.0*sqrt(nu)*nx12/abs2(t1) 
    a2_d2_d2 = besselk_dx_dx(nu, 2.0*sqrt(nu)*nx12/t1)*abs2(2.0*sqrt(nu)*nx12/abs2(t1))
    a2_d2_d2+= 2.0*besselk_dx(nu, 2.0*sqrt(nu)*nx12/t1)*2.0*sqrt(nu)*nx12/(t1^3)
    out = t0*(a1*a2_d2_d2 + 2.0*a1_d2*a2_d2 + a1_d2_d2*a2)
  end
  return out
end

##
#
# Profiled new Stein matern with fixed nu:
#
##

function ps1_kernfun(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nu    = 1.0
  t0,t1 = 1.0, parms[1]
  nx12  = norm(x1-x2)
  out   = t0*(abs2(t1)/(4.0*nu))*(gamma(nu)*(2.0^(nu-1.0)))
  if nx12 != 0.0
    a1  = (nx12*t1/(2.0*sqrt(nu)))^nu
    a2  = besselk(nu, 2.0*sqrt(nu)*nx12/t1)
    out = t0*a1*a2
  end
  return out
end

function ps1_kernfun_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nu    = 1.0
  t0,t1 = 1.0, parms[1]
  nx12  = norm(x1-x2)
  out   = t0*2.0*(t1/(4.0*nu))*(gamma(nu)*(2.0^(nu-1.0)))
  if nx12 != 0.0
    a1    = (nx12*t1/(2.0*sqrt(nu)))^nu
    a1_d2 = (nu*(nx12*t1/(2.0*sqrt(nu)))^(nu-1.0))*(nx12/2.0*sqrt(nu))
    a2    = besselk(nu, 2.0*sqrt(nu)*nx12/t1)
    a2_d2 = -besselk_dx(nu, 2.0*sqrt(nu)*nx12/t1)*2.0*sqrt(nu)*nx12/abs2(t1) 
    out = t0*(a1*a2_d2 + a1_d2*a2)
  end
  return out
end

function ps1_kernfun_d2_d2(x1::AbstractVector, x2::AbstractVector, parms::AbstractVector)
  nu    = 1.0
  t0,t1 = 1.0, parms[1]
  nx12  = norm(x1-x2)
  out   = t0*(2.0/(4.0*nu))*(gamma(nu)*(2.0^(nu-1.0)))
  if nx12 != 0.0
    a1       = (nx12*t1/(2.0*sqrt(nu)))^nu
    a1_d2    = (nu*(nx12*t1/(2.0*sqrt(nu)))^(nu-1.0))*(nx12/2.0*sqrt(nu))
    a1_d2_d2 = (nu*(nu-1.0)*(nx12*t1/(2.0*sqrt(nu)))^(nu-2.0))*abs2(nx12/(2.0*sqrt(nu)))
    a2       = besselk(nu, 2.0*sqrt(nu)*nx12/t1)
    a2_d2    = -besselk_dx(nu, 2.0*sqrt(nu)*nx12/t1)*2.0*sqrt(nu)*nx12/abs2(t1) 
    a2_d2_d2 = besselk_dx_dx(nu, 2.0*sqrt(nu)*nx12/t1)*abs2(2.0*sqrt(nu)*nx12/abs2(t1))
    a2_d2_d2+= 2.0*besselk_dx(nu, 2.0*sqrt(nu)*nx12/t1)*2.0*sqrt(nu)*nx12/(t1^3)
    out = t0*(a1*a2_d2_d2 + 2.0*a1_d2*a2_d2 + a1_d2_d2*a2)
  end
  return out
end


