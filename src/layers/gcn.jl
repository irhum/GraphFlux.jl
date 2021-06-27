function indegrees(receivers, bins)
    ones = (similar(receivers) .= 1)
    scatter(ones, receivers, bins, +)
end

function indegrees(g)
    indegrees(receivers(g), nnodes(g))
end

Flux.@nograd function symmetricnorm(g, indegrees)
    d = max.(indegrees, 1)

    dᵢ = gather(d, senders(g))
    dⱼ = gather(d, receivers(g))

    # TODO: actual solution for type stability!
    Float32.(sqrt.(1 ./ dᵢ) .* sqrt.(1 ./ dⱼ))
end

function symmetricnorm(g)
    indegrees = indegrees(receivers(g), nnodes(g))
    symmetricnorm(g, indegrees)
end

Flux.@nograd indegrees
Flux.@nograd symmetricnorm

struct GCN
    l
end

function GCN(in::Integer, out::Integer, σ=identity)
    return GCN(Dense(in, out, σ))
end

function (layer::GCN)(g::AbstractGraphTuple, symnorm::Bool=true)
    l = layer.l

    gathered = gather(nodes(g), senders(g))
    if symnorm gathered = symmetricnorm(g) .* gathered end

    l(scatter(gathered, receivers(g), nnodes(g), +))
end

Flux.@functor GCN

struct GCNₑ
    l
    edgeembedding
    rootembedding
end

function GCNₑ(in::Integer, inₑ::Integer, out::Integer, σ=identity)
    l = Dense(in, out, σ)
    edgeembedding = Flux.glorot_uniform(out, inₑ)
    rootembedding = randn(out)
    return GCNₑ(l, edgeembedding, rootembedding)
end

using Zygote 

function (layer::GCNₑ)(g::AbstractGraphTuple)
    nodeh = layer.l(nodes(g))
    edgeh = layer.edgeembedding * edges(g)
    
    deg = indegrees(g) .+ 1

    gathered = gather(nodeh, senders(g))
    gathered = reshape(symmetricnorm(g, deg), 1, :) .* relu.(gathered .+ edgeh)

    nodeh = scatter(gathered, receivers(g), nnodes(g), +) .+ relu.(nodeh .+ layer.rootembedding) ./ reshape(deg, 1, :)
    # TODO: make this NOT state changing
    updatenodes(g, nodeh)
end

Flux.@functor GCNₑ 
