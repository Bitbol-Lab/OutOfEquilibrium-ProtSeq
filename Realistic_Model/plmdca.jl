using PlmDCA
using JLD
using NPZ
using LinearAlgebra
filename = ARGS[1]
outputfilename = ARGS[2]
print(filename)
msa = transpose(npzread(filename))
msa = (msa .+ 1)
msa = convert(Array{Int8}, msa)
nbrpos, nbrseq = size(msa)
W = ones(nbrseq)/nbrseq
X_default = plmdca_asym(msa, W,remove_dups = false, verbose = true)
fname, _ = split(filename, ".npy")
save(outputfilename,"couplings",X_default.Jtensor)