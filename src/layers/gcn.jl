struct GCN
    W::AbstractMatrix
    σ::Function
end

function GCN(in::Integer, out::Integer, σ=identity)
    return GCN(rand(out, in), σ)
end

function (layer::GCN)(Xₙ, P)
    W, σ = layer.W, layer.σ
    return σ.(W * Xₙ * P)
end

Flux.@functor GCN 
