abstract type AbstractGraphTuple end
mutable struct GraphTuple <: AbstractGraphTuple
    nodes::Dict
    edges::Dict
    globals::Dict
    senders::AbstractVector{<:Integer}
    receivers::AbstractVector{<:Integer}
end

arrdict = Dict{Any, AbstractArray}
function GraphTuple(;nodes=arrdict(), edges=arrdict(), globals=arrdict(),
                    senders, receivers)

    if typeof(nodes) <: AbstractArray nodes=arrdict("default" => nodes) end
    if typeof(edges) <: AbstractArray edges=arrdict("default" => edges) end
    if typeof(globals) <: AbstractArray globals=arrdict("default" => globals) end

    @assert length(senders) == length(receivers)

    return GraphTuple(nodes, edges, globals, senders, receivers)
end

nodes(g::GraphTuple, key="default") = g.nodes[key]
edges(g::GraphTuple, key="default") = g.edges[key]
globals(g::GraphTuple, key="default") = g.globals[key]

senders(g::GraphTuple) = g.senders
receivers(g::GraphTuple) = g.receivers

nnodes(g::GraphTuple) = size(nodes(g), 2)
nedges(g::GraphTuple) = g |> senders |> length

updatenodes!(g::GraphTuple, v, key="default") = g.nodes[key] = v
updateedges!(g::GraphTuple, v, key="default") = g.edges[key] = v
updateglobals!(g::GraphTuple, v, key="default") = g.globals[key] = v



