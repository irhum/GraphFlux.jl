abstract type AbstractGraphTuple end
mutable struct GraphTuple <: AbstractGraphTuple
    nodes
    edges
    globals
    senders::AbstractVector{<:Integer}
    receivers::AbstractVector{<:Integer}
end

function GraphTuple(;nodes=nothing, edges=nothing, globals=nothing,
                    senders, receivers)
    
    @assert length(senders) == length(receivers)

    return GraphTuple(nodes, edges, globals, senders, receivers)
end

nodes(g::AbstractGraphTuple) = g.nodes
edges(g::AbstractGraphTuple) = g.edges
globals(g::AbstractGraphTuple) = g.globals

senders(g::AbstractGraphTuple) = g.senders
receivers(g::AbstractGraphTuple) = g.receivers

# TODO: does this fully make sense?
nnodes(g::AbstractGraphTuple) = size(nodes(g), 2)
nedges(g::AbstractGraphTuple) = g |> senders |> length

updatenodes(g::AbstractGraphTuple, v) = @set g.nodes = v
updateedges(g::AbstractGraphTuple, v) = @set g.edges = v
updateglobals(g::AbstractGraphTuple, v) = @set g.globals = v

mutable struct BatchGraphTuple <: AbstractGraphTuple
    nodes
    edges
    globals
    senders::AbstractVector{<:Integer}
    receivers::AbstractVector{<:Integer}

    node2graph::AbstractVector{<:Integer}
    edge2graph::AbstractVector{<:Integer}
    numgraphs::Integer
end

function batchgraphs(gs::Vector{GraphTuple}, nodespergraph, edgespergraph, symmetrize=false)
    nodesv = hcat([nodes(g) for g in gs]...)
    edgesv = hcat([edges(g) for g in gs]...)
    globalsv = hcat([globals(g) for g in gs]...)

    numgraphs = length(gs)
    nodes2graph = vcat(fill.(1:numgraphs, nodespergraph)...)
    edges2graph = vcat(fill.(1:numgraphs, edgespergraph)...)

    shift = circshift(cumsum(nodespergraph), 1)
    shift[1] = 0
    edgeshift = vcat(fill.(shift, edgespergraph)...)

    sendersv = vcat([senders(g) for g in gs]...) .+ edgeshift
    receiversv = vcat([receivers(g) for g in gs]...) .+ edgeshift

    if symmetrize
        sendersvsym = vcat([sendersv, receiversv]...)
        receiversvsym = vcat([receiversv, sendersv]...)
        sendersv, receiversv = sendersvsym, receiversvsym

        edges2graph = repeat(edges2graph, 2)
        edgesv = repeat(edgesv, 1, 2)
    end

    return BatchGraphTuple(nodesv, edgesv, globalsv,
                            sendersv, receiversv,
                            nodes2graph, edges2graph, numgraphs)
end


# better approach than overriding existing types?
# function Functors.functor(d::Dict{T, <:AbstractArray}) where {T}
#     kvs = [(k, v) for (k, v) in d]
#     ks = [k for (k, _) in kvs]
#     vs = [v for (_, v) in kvs]
    
#     function reconstruct(all)
#         Dict(k=>v for (k, v) in zip(ks, all.vs))
#     end
#     return (vs=vs,), reconstruct
# end


Flux.@functor BatchGraphTuple