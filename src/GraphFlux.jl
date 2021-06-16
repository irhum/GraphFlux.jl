module GraphFlux

using CUDA

import Base
import CUDA.CUSPARSE: AbstractCuSparseMatrix
import SparseArrays: dimlub

export 
    CuSparseMatrixHCOO

include("cuda/hcoo.jl")
include("cuda/ops/scatter.jl")
include("cuda/ops/gather.jl")

end
