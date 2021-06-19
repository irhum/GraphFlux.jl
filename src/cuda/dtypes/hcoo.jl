# Defining the type
mutable struct CuSparseMatrixHCOO{Tv,N} <: AbstractCuSparseMatrix{Tv}
    rowInd::CuVector{<:Int}
    colInd::CuVector{<:Int}
    nzVal::CuArray{Tv, N}
    dims::NTuple{2,Int}
    nnz::Int
end

function CuSparseMatrixHCOO(rowInd, colInd, nzVal, dims)
    CuSparseMatrixHCOO(rowInd,colInd,nzVal,dims,length(nzVal))
end

function CuSparseMatrixHCOO(rowInd, colInd, nzVal)
    CuSparseMatrixHCOO(rowInd,colInd,nzVal,(dimlub(rowInd),dimlub(colInd)),length(nzVal))
end

Base.:size(A::CuSparseMatrixHCOO) = A.dims

# Defining the fused message-aggregate operation
function Base.:*(X::CuMatrix, A::CuSparseMatrixHCOO, ⊕)
    function kernel!(O, X, I, J, V)
        thread = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
        i = thread % size(O, 1) + 1
        j = thread ÷ size(O, 1) + 1
    
        @inbounds if i <= size(O, 1) && j <= length(J)
            @atomic O[i, J[j]] = ⊕(O[i, J[j]], X[i, I[j]] * V[j])
        end
        return
    end
    
    # TODO: FIX TYPES
    O = CUDA.zeros(size(X, 1), size(A, 2))

    threads = 256
    blocks = ceil(Int, size(O, 1) * length(A.nzVal) / threads)
    @cuda blocks=blocks threads=threads kernel!(O, X, A.rowInd, A.colInd, A.nzVal)

    return O
end

Base.:*(X::CuMatrix, A::CuSparseMatrixHCOO) = *(X, A, +)

function Base.show(io::IO, A::CuSparseMatrixHCOO)
    print(io, "CuSparseMatrixHCOO of size $size(A) with $(A.nnz) non-zero elements")
end