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
@time tup = GraphFlux.batch(repeat([tup1], 4000), repeat([19], 4000), repeat([20], 4000))

gcn = GCNₑ(173, 13, 300) |> gpu
gcn2 = GCNₑ(300, 13, 300) |> gpu

using Zygote

tupG = tup |> gpu

ps = Params(Flux.params(gcn, gcn2))

b = Zygote.gradient(ps) do 
        a = tupG |> gcn |> gcn2 |> gcn2
        sum(GraphFlux.nodes(a))
    end

b[ps[1]]
using BenchmarkTools
@benchmark (CUDA.@sync begin
    b2 = Zygote.gradient(ps) do 
        a = tupG |> gcn |> gcn2 |> gcn2
        # a = tup |> gcn
        sum(GraphFlux.nodes(a))
    end
end) seconds=0.5

@benchmark (CUDA.@sync begin
        a = tupG |> gcn |> gcn2 |> gcn2
        sum(GraphFlux.nodes(a))
end) seconds=0.5

c = Array(1:24)


using Random
randperm(200)

