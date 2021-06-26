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
