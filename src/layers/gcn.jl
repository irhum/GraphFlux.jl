function indegrees(receivers, bins)
    ones = (similar(receivers) .= 1)
    scatter(ones, receivers, bins, +)
end

function indegrees(g)
    indegrees(receivers(g), nnodes(g))
end

function symmetricnorm(g, indegreesv)
    d = max.(indegreesv, 1)

    dᵢ = gather(d, senders(g))
    dⱼ = gather(d, receivers(g))

    # TODO: actual solution for type stability!
    Float32.(sqrt.(1 ./ dᵢ) .* sqrt.(1 ./ dⱼ))
end

function symmetricnorm(g)
    indegreesv = indegrees(receivers(g), nnodes(g))
    symmetricnorm(g, indegreesv)
end

function nodespergraph(g::BatchGraphTuple)
    ones = similar(g.nodes, Int32, size(g.nodes, 2))
    ones .= 1
    scatter(ones, g.node2graph, g.numgraphs, +)
end

Flux.@nograd indegrees
Flux.@nograd symmetricnorm
Flux.@nograd nodespergraph

struct GCN
    l
    σ
end

function GCN(in::Integer, out::Integer, σ=identity)
    return GCN(Dense(in, out, bias=false), σ)
end

function (l::GCN)(nodes, senders, receivers; symnorm::Bool=true)
    gathered = gather(nodes, senders)
    if symnorm gathered = symmetricnorm(g) .* gathered end
    l.σ.(l.l(scatter(gathered, receivers, size(nodes, 2), +)))
end

function (l::GCN)(g::AbstractGraphTuple; symnorm::Bool=true)
    updatenodes(g, l(g.nodes, g.senders, g.receivers, symnorm=symnorm))
end

Flux.@functor GCN


function graphmeanpool(g::BatchGraphTuple)
    nodesperg = nodespergraph(g)
    scatter(g.nodes, g.node2graph, g.numgraphs, +) ./ Flux.unsqueeze(nodesperg, 1)
end