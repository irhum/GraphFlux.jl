module GraphFlux

using Base
using CUDA
using Flux 
using Functors
using ChainRulesCore
using Setfield

import Base
import CUDA.CUSPARSE: AbstractCuSparseMatrix
import SparseArrays: dimlub

export 
    CuSparseMatrixHCOO,

    GraphTuple, 
    BatchGraphTuple,
    batchgraphs,

    GCN,
    GCNₑ,
    graphmeanpool,

    OGBAtomEncoder

include("dtypes/graphtuple.jl")

include("ops/messageprop.jl")

# GPU ops
include("cuda/ops/messageprop.jl")

# layers 
include("layers/gcn.jl")

# features
include("features/molecules/ogbhiv.jl")

end
