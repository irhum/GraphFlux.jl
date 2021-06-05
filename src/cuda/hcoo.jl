using CUDA
using Base 

import CUDA.CUSPARSE: AbstractCuSparseMatrix
import SparseArrays: dimlub

mutable struct CuSparseMatrixHCOO{Tv,N} <: AbstractCuSparseMatrix{Tv}
    rowInd::CuVector{<:Int}
    colInd::CuVector{<:Int}
    nzVal::CuArray{Tv, N}
    dims::NTuple{2,Int}
    nnz::Int
end

function CuSparseMatrixHCOO(rowInd, colInd, nzVal, dims)
    CuSparseMatrixHCOO(rowInd,colInd,nzVal,dims,size(nzVal, 1))
end

function CuSparseMatrixHCOO(rowInd, colInd, nzVal)
    CuSparseMatrixHCOO(rowInd,colInd,nzVal,(dimlub(rowInd),dimlub(colInd)),length(nzVal))
end

Base.:size(A::CuSparseMatrixHCOO) = A.dims

n = 5000
nz = 30
dim = (n, n)

x = rand(1:n, nz)
y = rand(1:n, nz)
v = rand(Float32, nz)

X = x |> cu
Y = y |> cu
V = v |> cu

@time ARR1 = CuSparseMatrixHCOO(X, Y, V, dim, nz)
@time ARR2 = CuSparseMatrixHCOO(X, Y, V, dim)
@time ARR3 = CuSparseMatrixHCOO(X, Y, V)
print(":)")
Int <: Integer

ARR2 = CuSparseMatrixHCOO(X, Y, repeat(V, 1, 3, 2), dim)
repeat(V, 1, 3)
[1, 2, 3][UInt(2)]