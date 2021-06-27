using BSON 
using Flux 

using Revise
using GraphFlux
using Random


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

function batch(features, graphdata, idxs)
    cumnumnodes = cumsum(graphdata["numnodes"])
    cumnumedges = cumsum(graphdata["numedges"])

    graphs = Vector{GraphTuple}(undef, length(idxs))

    for (i, idx) in enumerate(idxs)
        nodef = features["node"][:, (idx == 1 ? 1 : cumnumnodes[idx - 1]+1):cumnumnodes[idx]]
        nodef = nhotfeats(nodef, NUM_ATOM_FEATURES)

        edgeidxs = (idx == 1 ? 1 : cumnumedges[idx - 1]+1):cumnumedges[idx]
        edgef = nhotfeats(features["edge"][:, edgeidxs], NUM_BOND_FEATURES)

        senders = graphdata["edgelist"][edgeidxs, 1]
        receivers = graphdata["edgelist"][edgeidxs, 2]

        graphs[i] = GraphTuple(nodes=nodef, edges=edgef, senders=senders, receivers=receivers)
    end
    batchgraphs(graphs, graphdata["numnodes"][idxs], graphdata["numedges"][idxs])
end

graph = batch(features, graphdata, idxs["test"])

gcn = GCNₑ(173, 13, 300) |> gpu
gcn2 = GCNₑ(300, 13, 300) |> gpu

graphG = graph |> gpu

ps = Flux.params(gcn, gcn2)

using BenchmarkTools
using CUDA 

b = Flux.gradient(ps) do 
        a = graphG |> gcn |> gcn2 |> gcn2
        sum(GraphFlux.nodes(a))
    end

@benchmark (CUDA.@sync begin
    b = Flux.gradient(ps) do 
        a = graphG |> gcn |> gcn2 |> gcn2 |> gcn2 |> gcn2
        sum(GraphFlux.nodes(a))
    end
end) seconds=1.25
