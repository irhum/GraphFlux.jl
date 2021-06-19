module GraphFlux

using CUDA
using Flux 

import Base
import CUDA.CUSPARSE: AbstractCuSparseMatrix
import SparseArrays: dimlub

export 
    CuSparseMatrixHCOO,

    GCN

include("cuda/dtypes/hcoo.jl")


include("ops/messageprop.jl")
include("ops/symmetrize.jl")

# GPU ops
include("cuda/ops/messageprop.jl")

# layers 
include("layers/gcn.jl")

end
