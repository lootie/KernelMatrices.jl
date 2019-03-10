
using LinearAlgebra, StaticArrays, Rsvg, Plots, JLD, LaTeXStrings, ColorSchemeTools
pyplot()

data     = load("../../../data/likelihood_surfaces_rank.jld")
surfaces = data["srfs"]
trup     = data["trup"]
sprd     = data["sprd"]
gdsz     = data["gdsz"]
gd1      = LinRange(trup[1] - sprd, trup[1] + sprd, gdsz)
gd2      = LinRange(trup[2] - sprd, trup[2] + sprd, gdsz)
ranks    = data["rnks"]


# Make the color scheme:
cfn(x)  = 1.0 - x^0.45
carg    = cgrad(make_colorscheme(cfn, cfn, cfn).colors)
clval   = 30.0
lvls    = [0.0, 0.0125, 0.025, 0.0375, 0.05, 0.075, 0.125, 0.2, 0.25, 0.4, 0.75, 1.0].*clval

function getmle(M::Matrix{Float64})
  return collect(collect(Iterators.product(gd1, gd2))[findmin(M)[2]])
end

function plot_rnk_surface(M::Matrix{Float64}, rnk::Int64, mle::Vector{Float64}, ceilval::Float64=clval)
  treated = map(x->min(x, ceilval), M .- minimum(M))
  if rnk == 2
    pt = contour(gd2, gd1, treated, title="Rank: "*string(rnk)*", min:"*string(round(minimum(M),digits=3)),
                 titlefont=font(11), tickfont=font(11), xticks=[4.8, 4.9, 5.0, 5.1, 5.2],
                 grid=false, size=(1500,375).*0.75, color=carg, fill=true, colorbar=false, levels=lvls)
  elseif rnk == 0
    pt = contour(gd2, gd1, treated, 
                 title="Exact, min:"*string(round(minimum(M),digits=3)),
                 titlefont=font(11), tickfont=font(11), xticks=[4.8, 4.9, 5.0, 5.1, 5.2],
                 size=(1500,375).*0.75, color=carg, grid=false, fill=true, yticks=nothing, levels=lvls)
  else
    treated = map(x->min(x, ceilval), M .- minimum(M))
    pt = contour(gd2, gd1, treated, title="Rank: "*string(rnk)*", min:"*string(round(minimum(M),digits=3)),
                 titlefont=font(11), tickfont=font(11), yticks=nothing,
                 xticks=[4.8, 4.9, 5.0, 5.1, 5.2], colorbar=false,  grid=false,
                 size=(1500,375).*0.75, color=carg, fill=true, levels=lvls)
  end
  scatter!([trup[2]], [trup[1]], color=:cyan, label="", markersize=6, markershape=:xcross)
  scatter!([mle[2]], [mle[1]], color=:red, label="", markersize=6, markershape=:circle)
  return pt
end

ar1    = surfaces[[2,4,6, 1]]
ar2    = ranks[[2,4,6, 1]]
ar3    = getmle.(ar1)
out    = map(x->plot_rnk_surface(x[1], x[2], x[3]), zip(ar1, ar2, ar3))
 

# Save the figure:
Plots.savefig(plot(out..., layout=(1,4)), "../../../report/figures/loglik_surfaces_rank.eps")
