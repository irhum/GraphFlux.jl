abstract type AbstractGraphTuple end
mutable struct GraphTuple <: AbstractGraphTuple
    nodesdict::Dict
    edgesdict::Dict
    globalsdict::Dict
    senders::AbstractVector{<:Integer}
    receivers::AbstractVector{<:Integer}
end

arrdict = Dict{Any, AbstractArray}
function graphdefaults(nodes, edges, globals)
    if typeof(nodes) <: AbstractArray nodes=arrdict("default" => nodes) end
    if typeof(edges) <: AbstractArray edges=arrdict("default" => edges) end
    if typeof(globals) <: AbstractArray globals=arrdict("default" => globals) end

    return nodes, edges, globals
end

function GraphTuple(;nodes=arrdict(), edges=arrdict(), globals=arrdict(),
                    senders, receivers)
    
    @assert length(senders) == length(receivers)
    nodes, edges, globals = graphdefaults(nodes, edges, globals)

    return GraphTuple(nodes, edges, globals, senders, receivers)
end

nodes(g::AbstractGraphTuple, key="default") = g.nodesdict[key]
edges(g::AbstractGraphTuple, key="default") = g.edgesdict[key]
globals(g::AbstractGraphTuple, key="default") = g.globalsdict[key]

senders(g::AbstractGraphTuple) = g.senders
receivers(g::AbstractGraphTuple) = g.receivers

# TODO: does this fully make sense?
nnodes(g::AbstractGraphTuple) = size(nodes(g), 2)
nedges(g::AbstractGraphTuple) = g |> senders |> length

Flux.@nograd function updatenodes(g::AbstractGraphTuple, v, key="default")
    g = copy(g)
    d = copy(g.nodesdict)
    d[key] = v
    g.nodesdict = d
    return g
end

Flux.@nograd function updateedges(g::AbstractGraphTuple, v, key="default")
    g = copy(g)
    d = copy(g.edgesdict)
    d[key] = v
    g.edgesdict = d
    return g
end

Flux.@nograd function updateglobals(g::AbstractGraphTuple, v, key="default")
    g = copy(g)
    d = copy(g.globalsdict)
    d[key] = v
    g.globalsdict = d
    return g
end

mutable struct BatchGraphTuple <: AbstractGraphTuple
    nodesdict::Dict
    edgesdict::Dict
    globalsdict::Dict
    senders::AbstractVector{<:Integer}
    receivers::AbstractVector{<:Integer}

    node2graph::AbstractVector{<:Integer}
    edge2graph::AbstractVector{<:Integer}
    numgraphs::Integer
end

function batch(gs::Array{GraphTuple}, nodespergraph, edgespergraph)
    nodesdict = Dict(k=>hcat([nodes(g, k) for g in gs]...) for (k,_) in gs[1].nodesdict)
    edgesdict = Dict(k=>hcat([edges(g, k) for g in gs]...) for (k,_) in gs[1].edgesdict)
    globaldict = Dict(k=>hcat([globals(g, k) for g in gs]...) for (k,_) in gs[1].globalsdict)

    numgraphs = length(gs)
    nodes2graph = vcat(fill.(1:numgraphs, nodespergraph)...)
    edges2graph = vcat(fill.(1:numgraphs, edgespergraph)...)

    shift = circshift(cumsum(nodespergraph), 1)
    shift[1] = 0
    edgeshift = vcat(fill.(shift, edgespergraph)...)

    # @show edgeshift[1:100]
    sendersv = vcat([senders(g) for g in gs]...) .+ edgeshift
    receiversv = vcat([receivers(g) for g in gs]...) .+ edgeshift

    return BatchGraphTuple(nodesdict, edgesdict, globaldict,
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

Base.copy(g::BatchGraphTuple) = BatchGraphTuple(g.nodesdict, g.edgesdict, g.globalsdict,
                                                g.senders, g.receivers,
                                                g.node2graph, g.edge2graph, g.numgraphs)
# Flux.@functor BatchGraphTuple