
These are the exact scripts used to generate the various results/benchmarks in the paper
corresponding with the release of this software suite.

Other than changing the path pointing to the relevant JLD files, you should not need to alter any of
the script to get the exact numerical results that we produce in the paper. 

For the sake of brevity, I used less verbose declarations of the boilerplate than I do in
../fitting/test_fit_*.jl. If some of the declarations aren't clear, like the HODLR.Maxlikopts(...)
struct, I suggest going back to those examples in ../fitting/, which are very verbose and pretty
exhaustive in explaining the options.

The directory ./figures/ shows the scripts used to generate the figures and tables used in the
paper. If you do run one of the scripts here and are interested in re-generating the relevant figure
or table, all you should need to do is change the path to the saved output file.

