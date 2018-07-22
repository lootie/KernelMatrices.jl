
using Plots, JLD, LaTeXStrings
pyplot()

data     = load("/path/to/likelihood_surfaces_rank.jld")
trup     = data["trup"]
sprd     = data["sprd"]
gdsz     = data["gdsz"]
gd1      = linspace(trup[1] - sprd , trup[1] + sprd, gdsz)
gd2      = linspace(trup[2] - sprd , trup[2] + sprd, gdsz)
ranks    = data["ranks"]

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

difm   = val_at_other_minimizers(data["srfs"])
diffs  = heatmap(vcat("X", string.(ranks)), 
                 vcat("X", string.(ranks)), 
                 difm, 
                 tickfont=font(18),
                 size=(500, 500)
                )

