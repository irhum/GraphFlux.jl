struct GCN
    W::AbstractMatrix
    σ::Function
end

function GCN(in::Integer, out::Integer, σ=identity)
    return GCN(rand(out, in), σ)
end

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

function (layer::GCN)(g::GraphTuple, symnorm::Bool=true)
    W, σ = layer.W, layer.σ

    gathered = gather(nodes(g), senders(g))

    if symnorm
        gathered = symmetricnorm(g) .* gathered
    end

    σ.(W * scatter(gathered, receivers(g), nnodes(g), +))
end

Flux.@functor GCN 
