module GraphFlux

using CUDA
using Flux 

import Base
import CUDA.CUSPARSE: AbstractCuSparseMatrix
import SparseArrays: dimlub

export 
    CuSparseMatrixHCOO,

    GCN

include("cuda/hcoo.jl")

# ops
include("cuda/ops/scatter.jl")
include("cuda/ops/gather.jl")

# layers 
include("layers/gcn.jl")

end
