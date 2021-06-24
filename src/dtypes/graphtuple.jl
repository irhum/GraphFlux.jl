abstract type AbstractGraphTuple end

Base.@kwdef struct GraphTuple <: AbstractGraphTuple
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

    return GraphTuple(nodes, edges, globals, senders, receivers)
end

nodes(g::GraphTuple, key="default") = g.nodes[key]
edges(g::GraphTuple, key="default") = g.edges[key]
globals(g::GraphTuple, key="default") = g.globals[key]
