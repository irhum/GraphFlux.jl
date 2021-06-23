module GraphFlux

using Base
using CUDA
using Flux 

import Base
import CUDA.CUSPARSE: AbstractCuSparseMatrix
import SparseArrays: dimlub

export 
    CuSparseMatrixHCOO,

    GCN

include("dtypes/graphtuple.jl")

include("ops/messageprop.jl")

# GPU ops
include("cuda/ops/messageprop.jl")

# layers 
include("layers/gcn.jl")

end
