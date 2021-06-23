struct GCN
    W::AbstractMatrix
    σ::Function
end

function GCN(in::Integer, out::Integer, σ=identity)
    return GCN(rand(out, in), σ)
end

function indegrees(targets, bins)
    vals = similar(targets, Int64, (1, length(targets)))
    vals .= 1

    reshape(scatter(vals, targets, bins, +), bins)
end

function symmetricnorm(g)
    d = max.(indegrees(g.receivers, size(g.nodes, 2)), 1)

    dᵢ = gather(reshape(d, (1, size(g.nodes, 2))), g.senders)
    dⱼ = gather(reshape(d, (1, size(g.nodes, 2))), g.receivers)

    sqrt.(1 ./ dᵢ) .* sqrt.(1 ./ dⱼ) .* 2
end

function (layer::GCN)(g::GraphTuple, symnorm::Bool=true)
    W, σ = layer.W, layer.σ

    gathered = gather(g.nodes, g.senders)

    if symnorm
        gathered = symmetricnorm(g) .* gathered
    end

    σ.(W * scatter(gathered, g.receivers, size(g.nodes, 2), +))
end

Flux.@functor GCN 
