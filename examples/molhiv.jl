using BSON 
using Flux 

using Revise
using GraphFlux

# atomnum, chirality, degree, charge, numH, radicale, hybridization, isarom, isisring
const NUM_ATOM_FEATURES = [119, 4, 12, 12, 10, 6, 6, 2, 2]

# bondtype, bondstereo, isconjugated
const NUM_BOND_FEATURES = [5, 6, 2]

function nhotfeats(a::AbstractArray, numfeats::AbstractVector, normalize=false)
    batchsize = size(a, 2)
    sumnumfeats = circshift(cumsum(numfeats), 1)
    sumnumfeats[1] = 0

    out = similar(a, Float32, (sum(numfeats), batchsize)) .= 0

    for i in 1:size(a, 2)
        out[a[:, i] .+ sumnumfeats, i] .= 1
    end

    normalize ? out ./ length(numfeats) : out
end

BSON.@load "hiv.bson" edgelist edgefeats nodefeats nodeidx edgeidx

edges = edgelist[1:edgeidx[1], :] .+ 1
sources = edges[:, 1]
receivers = edges[:, 2]

edgeh = nhotfeats(edgefeats[:, 1:edgeidx[1]], NUM_BOND_FEATURES)
nodeh = nhotfeats(nodefeats[:, 1:nodeidx[1]], NUM_ATOM_FEATURES) 

gcn = GCNₑ(173, 13, 300)

tup1 = GraphFlux.GraphTuple(nodes=nodeh, edges=edgeh, senders=sources, receivers=receivers)
@time tup = GraphFlux.batch(repeat([tup1], 80), repeat([19], 80), repeat([20], 80))
@time a = gcn(tup)

using Zygote

gcn = GCN(173, 300)
Zygote.gradient(l -> tup |> x -> gcn(x, false) |> sum, gcn)


Flux.params(tup)