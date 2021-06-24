function gather(data::AbstractMatrix, sources::AbstractVector)
    return data[:, sources]
end

function gather(data::AbstractVector, sources::AbstractVector)
    return data[sources]
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
    scatter(data, targets, bins, ⊕)
end