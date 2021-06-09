module GraphFlux

using CUDA

import Base
import CUDA.CUSPARSE: AbstractCuSparseMatrix
import SparseArrays: dimlub

export 
    CuSparseMatrixHCOO

include("cuda/hcoo.jl")

end
