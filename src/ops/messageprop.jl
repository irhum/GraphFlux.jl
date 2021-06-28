# inspired by pytorch geometric's message passing implementation
# https://arxiv.org/abs/1903.02428
function gather(data::AbstractMatrix, sources::AbstractVector)
    return data[:, sources]
end

function gather(data::AbstractVector, sources::AbstractVector)
    data = reshape(data, (1, length(data)))
    gather(data, sources)
end


function scatter(data::AbstractMatrix, targets::AbstractVector, bins, ⊕)
    output = similar(data, (size(data, 1), bins))
    output .= 0
    
    for i in 1:length(targets)
        output[:, targets[i]] = ⊕(output[:, targets[i]], data[:, i])
    end

    return output
end

function scatter(data::AbstractVector, targets::AbstractVector, bins, ⊕)
    data = reshape(data, (1, length(data)))
    scatter(data, targets, bins, ⊕)[1, :]
end


# derivatives!
# TODO: correct gradients for NON addition aggregation
function ChainRulesCore.rrule(::typeof(gather), data::AbstractMatrix, sources::AbstractVector)
    pullback(Ω̄) = (NoTangent(), scatter(Ω̄, sources, size(data, 2), +), ZeroTangent())
    return gather(data, sources), pullback
end

function ChainRulesCore.rrule(::typeof(scatter), data::AbstractMatrix, targets::AbstractVector, bins, ⊕)
    pullback(Ω̄) = (NoTangent(), gather(Ω̄, targets), ZeroTangent(), ZeroTangent(), ZeroTangent())
    return scatter(data, targets, bins, ⊕), pullback
end