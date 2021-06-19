function indegrees(targets, bins)
    vals = similar(targets, Int64, (1, length(targets)))
    vals .= 1

    reshape(scatter(vals, targets, bins, +), bins)
end

function symmetricnorm(I::AbstractVector, J::AbstractVector, dims::NTuple{2, Int})
    d = indegrees(J, dims[2])

    dᵢ = gather(reshape(d, (1, dims[2])), I)
    dⱼ = gather(reshape(d, (1, dims[2])), J)

    reshape(sqrt.(1 ./ dᵢ) .* sqrt.(1 ./ dⱼ) .* 2, length(I))
end