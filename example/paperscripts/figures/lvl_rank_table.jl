
using JLD, StaticArrays

function val_at_other_minimizer(V::Vector{Matrix{Float64}}, j::Int64, k::Int64)
  return abs(V[j][indmin(V[j])] - V[j][indmin(V[k])])
end

# Prepare the two tables to mutate:
tables = map(x->zeros(Float64, 2, 6), 1:2)

# Loop over the two files:
for (j, data_path) in enumerate(["/path/to/likelihood_surfaces.jld",
                                 "/path/to/likelihood_surfaces_rank.jld"])
  # Load in the data file:
  data     = load(data_path)
  # Get the two rows of interest:
  tables[j][1,:] = map(k -> val_at_other_minimizer(data["srfs"], 1, k), 1:6)
  tables[j][2,:] = map(k -> val_at_other_minimizer(data["srfs"], k, 1), 1:6)
end

