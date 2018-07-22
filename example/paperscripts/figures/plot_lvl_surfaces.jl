
using Rsvg, Plots, JLD, LaTeXStrings
plotlyjs()

data     = load("/path/to/likelihood_surfaces.jld")
surfaces = data["srfs"]
trup     = data["trup"]
sprd     = data["sprd"]
gdsz     = data["gdsz"]
gd1      = linspace(trup[1] - sprd , trup[1] + sprd, gdsz)
gd2      = linspace(trup[2] - sprd , trup[2] + sprd, gdsz)

function minspt(M::Matrix{Float64})
  j,k=collect(Iterators.product(gd1, gd2))[indmin(M)]
  return j,k
end

function plot_lvl_surface(M::Matrix{Float64}, lvl::Union{String, Int64})
  if lvl == "level 5"
    pt = heatmap(gd2, gd1, M.-minimum(M), title=string(lvl)*" , min:"*string(round(minimum(M), 3)),
                 titlefont=font(11), tickfont=font(11), xticks=[4.8, 4.9, 5.0, 5.1, 5.2],
                 yticks=nothing, grid=false,
                 size=(1500,375).*0.75, clim=(0.0, 30.0))
  elseif lvl == "Exact"
    pt = heatmap(gd2, gd1, M.-minimum(M), title=string(lvl)*" , min:"*string(round(minimum(M), 3)),
                 titlefont=font(11), tickfont=font(11), xticks=[4.8, 4.9, 5.0, 5.1, 5.2],
                 size=(1500,375).*0.75, clim=(0.0, 30.0), colorbar=false, grid=false)
  else
    pt = heatmap(gd2, gd1, M.-minimum(M), title=string(lvl)*" , min:"*string(round(minimum(M), 3)),
                 titlefont=font(11), tickfont=font(11), yticks=nothing,
                 xticks=[4.8, 4.9, 5.0, 5.1, 5.2], colorbar=false,  grid=false,
                 size=(1500,375).*0.75, clim=(0.0, 30.0))
  end
  mj,mk = minspt(M)
  scatter!([trup[2]], [trup[1]], color=:cyan, label="", markersize=4)
  scatter!([mk], [mj], color=:red, label="", markersize=4)
  return pt
end

function val_at_other_minimizers(V::Vector{Matrix{Float64}})
  out  = zeros(length(V), length(V))
  mins = indmin.(V)
  for j in eachindex(V)
    for k in eachindex(V)
      out[j,k] = abs(V[j][mins[j]] - V[j][mins[k]])
    end
  end
  return out
end

# The two main plots:
ar1    = surfaces[[1,2,4,6]]
ar2    = vcat("Exact", map(x->"level "*string(x), 1:(length(surfaces)-1)))[[1,2,4,6]]
plots2 = map(x->plot_lvl_surface(x[1], x[2]), zip(ar1, ar2))
difm   = val_at_other_minimizers(surfaces)
diffs  = heatmap(vcat("X", string.(1:(length(surfaces)-1))),
                 vcat("X", string.(1:(length(surfaces)-1))), 
                 difm, 
                 tickfont=font(18),
                 size=(500, 500)
                )

