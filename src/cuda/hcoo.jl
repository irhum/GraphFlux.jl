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
nz = 30000
dim = (n, n)

i = rand(1:n, nz)
j = rand(1:n, nz)
v = rand(Float32, nz)

I = i |> cu
J = j |> cu
V = v |> cu

X = rand(32, n) |> cu

@time ARR1 = CuSparseMatrixHCOO(X, Y, V, dim, nz)
@time ARR2 = CuSparseMatrixHCOO(X, Y, V, dim)
@time ARR3 = CuSparseMatrixHCOO(X, Y, V)
print(":)")
Int <: Integer

ARR2 = CuSparseMatrixHCOO(X, Y, repeat(V, 1, 3, 2), dim)
repeat(V, 1, 3)
[1, 2, 3][UInt(2)]

using CUDA
using SparseArrays

function muldensesparsecoo(X::CuMatrix, I::CuVector, J::CuVector, V::CuVector)
    function kernel!(O, X, I, J, V)
        thread = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
        i = thread % size(O, 1) + 1
        j = thread ÷ size(O, 1) + 1
    
        @inbounds if i <= size(O, 1) && j <= length(J)
            @atomic O[i, J[j]] += X[i, I[j]] * V[j]
        end
        return
    end
   
    O = CUDA.zeros(X |> size)

    threads = 256
    blocks = ceil(Int, size(O, 1) * length(V) / threads)
    @cuda blocks=blocks threads=threads kernel!(O, X, I, J, V)

    return O
end

n = 5000
nz = 30000
dim = (n, n)

i = rand(1:n, nz)
j = rand(1:n, nz)
v = rand(Float32, nz)

I = i |> cu
J = j |> cu
V = v |> cu

X = rand(32, n) |> cu
A = sparse(i, j, v) |> Array |> cu

@time (CUDA.@sync O1 = muldensesparsecoo(X, I, J, V))
@time (CUDA.@sync O2 = X * A)

sum((O1 .- O2).^2)

X = rand(Float16, 32, n) |> cu
A = sparse(i, j, v) |> Array |> cu

@time (CUDA.@sync O1 = muldensesparsecoo(X, I, J, V))
@time (CUDA.@sync O2 = X * A)

sum((O1 .- O2).^2)