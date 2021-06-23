abstract type AbstractGraphTuple end

Base.@kwdef struct GraphTuple <: AbstractGraphTuple
    nodes=missing
    edges=missing
    globals=missing
    senders::AbstractVector{<:Integer}
    receivers::AbstractVector{<:Integer}
end
