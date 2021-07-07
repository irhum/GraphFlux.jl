module GraphFlux

using CUDA
using Flux 
using Functors
using ChainRulesCore
using Setfield

using DifferentialEquations
using DiffEqFlux

export 
    GraphTuple, 
    BatchGraphTuple,
    batchgraphs,

    GCN,
    GCNₑ,
    NodeLayer,
    graphmeanpool,

    GraphNeuralODE

include("dtypes/graphtuple.jl")

include("ops/messageprop.jl")

# GPU ops
include("cuda/ops/messageprop.jl")

# layers 
include("layers/gcn.jl")
include("layers/graphode.jl")

# ogbhiv
include("features/molecules/ogbhiv.jl")

end
