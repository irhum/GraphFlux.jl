function indegrees(receivers, bins)
    ones = (similar(receivers) .= 1)
    scatter(ones, receivers, bins, +)
end

function symmetricnorm(g)
    d = max.(indegrees(receivers(g), nnodes(g)), 1)

    dᵢ = gather(d, senders(g))
    dⱼ = gather(d, receivers(g))

    sqrt.(1 ./ dᵢ) .* sqrt.(1 ./ dⱼ) .* 2
end

struct GCN
    l
end

function GCN(in::Integer, out::Integer, σ=identity)
    return GCN(Dense(in, out, σ))
end

function (layer::GCN)(g::GraphTuple, symnorm::Bool=true)
    l = layer.l

    gathered = gather(nodes(g), senders(g))
    if symnorm gathered = symmetricnorm(g) .* gathered end

    l(scatter(gathered, receivers(g), nnodes(g), +))
end

Flux.@functor GCN 
