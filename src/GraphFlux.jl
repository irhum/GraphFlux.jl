module GraphFlux

using CUDA
using Flux 
using Functors
using ChainRulesCore
using Setfield

export 
    GraphTuple, 
    BatchGraphTuple,
    batchgraphs,

    GCN,
    GCNₑ,
    graphmeanpool

include("dtypes/graphtuple.jl")

include("ops/messageprop.jl")

# GPU ops
include("cuda/ops/messageprop.jl")

# layers 
include("layers/gcn.jl")

# ogbhiv
include("features/molecules/ogbhiv.jl")

end
