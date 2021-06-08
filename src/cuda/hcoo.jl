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
    CuSparseMatrixHCOO(rowInd,colInd,nzVal,dims,length(nzVal))
end

function CuSparseMatrixHCOO(rowInd, colInd, nzVal)
    CuSparseMatrixHCOO(rowInd,colInd,nzVal,(dimlub(rowInd),dimlub(colInd)),length(nzVal))
end

Base.:size(A::CuSparseMatrixHCOO) = A.dims

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

using CSV, DataFrames, CodecZlib, Mmap

A = CSV.File(transcode(GzipDecompressor, Mmap.mmap("hiv/raw/edge.csv.gz"))) |> DataFrame

A = CSV.File(transcode(GzipDecompressor, Mmap.mmap("hiv/raw/edge-feat.csv.gz"))) |> DataFrame
A = CSV.File(transcode(GzipDecompressor, Mmap.mmap("hiv/raw/node-feat.csv.gz"))) |> DataFrame
