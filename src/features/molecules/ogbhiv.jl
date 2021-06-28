# PREPROCESSING
# atomnum, chirality, degree, charge, numH, radicale, hybridization, isarom, isisring
const OGB_NUM_ATOM_FEATURES = [119, 4, 12, 12, 10, 6, 6, 2, 2]

# bondtype, bondstereo, isconjugated
const OGB_NUM_BOND_FEATURES = [5, 6, 2]

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

# dataloading
function getmolbatch(features, graphdata, idxs)
    cumnumnodes = cumsum(graphdata["numnodes"])
    cumnumedges = cumsum(graphdata["numedges"])

    graphs = Vector{GraphTuple}(undef, length(idxs))

    for (i, idx) in enumerate(idxs)
        nodef = features["node"][:, (idx == 1 ? 1 : cumnumnodes[idx - 1]+1):cumnumnodes[idx]]
        nodef = nhotfeats(nodef, OGB_NUM_ATOM_FEATURES)

        edgeidxs = (idx == 1 ? 1 : cumnumedges[idx - 1]+1):cumnumedges[idx]
        edgef = nhotfeats(features["edge"][:, edgeidxs], OGB_NUM_BOND_FEATURES)

        senders = graphdata["edgelist"][edgeidxs, 1]
        receivers = graphdata["edgelist"][edgeidxs, 2]

        graphs[i] = GraphTuple(nodes=nodef, edges=edgef, senders=senders, receivers=receivers)
    end
    batchgraphs(graphs, graphdata["numnodes"][idxs], graphdata["numedges"][idxs], true)
end

# ENCODERS
struct NHotEncoder
    W
end

function NHotEncoder(featdims::AbstractVector{<:Integer}, embdim::Integer)
    W = zeros(embdim, sum(featdims))

    current = 1
    for featdim in featdims
        W[:, current:current+featdim-1] .= Flux.glorot_uniform(embdim, featdim)
        current += featdim
    end

    return NHotEncoder(W)
end

(l::NHotEncoder)(nhotmatrix::AbstractMatrix) = l.W * nhotmatrix

Flux.@functor NHotEncoder 

# atom and bond encoders
struct OGBAtomEncoder 
    embedder 
end

OGBAtomEncoder(embdim::Integer) = OGBAtomEncoder(NHotEncoder(OGB_NUM_ATOM_FEATURES, embdim))
(l::OGBAtomEncoder)(g::AbstractGraphTuple) = @set g.nodes = l.embedder(g.nodes)
Flux.@functor OGBAtomEncoder

struct OGBBondEncoder 
    embedder 
end

OGBBondEncoder(embdim::Integer) = OGBBondEncoder(NHotEncoder(OGB_NUM_BOND_FEATURES, embdim))
(l::OGBBondEncoder)(g::AbstractGraphTuple) = @set g.edges = l.embedder(g.edges)
Flux.@functor OGBBondEncoder

# GNN LAYERS
struct GCNₑ
    l
    edgeembedding
    rootembedding
end

function GCNₑ(in::Integer, out::Integer, σ=identity)
    l = Dense(in, out, σ)
    edgeembedding = OGBBondEncoder(out)
    rootembedding = randn(out)
    return GCNₑ(l, edgeembedding, rootembedding)
end


function (layer::GCNₑ)(g::AbstractGraphTuple)
    edgeh = layer.edgeembedding(g) |> edges
    nodeh = layer.l(nodes(g))
    
    deg = indegrees(g) .+ 1

    gathered = gather(nodeh, senders(g))
    gathered = reshape(symmetricnorm(g, deg), 1, :) .* relu.(gathered .+ edgeh)

    nodeh = scatter(gathered, receivers(g), nnodes(g), +) .+ relu.(nodeh .+ layer.rootembedding) ./ reshape(deg, 1, :)

    updatenodes(g, nodeh)
end

Flux.@functor GCNₑ 
