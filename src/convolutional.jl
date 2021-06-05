using CUDA, CUDA.CUSPARSE
using SparseArrays 

n = 5000
nz = 30
dim = (n, n)

function ConvGNN(xᵢ, xⱼ, C, ⊕, ϕ, ψ)
    ϕ(xᵢ, C * )
end 
x = rand(1:n, nz)
y = rand(1:n, nz)
v = rand(Float32, nz)

X = convert(CuArray{Cint}, x)
Y = convert(CuArray{Cint}, y)
V = v |> cu

ARR = CuSparseMatrixCOO{Float32}(X, Y, V, dim)
B = CUDA.rand(5000, 20)

ARR2 = ARR |> CuSparseMatrixCSR
ARR2 * B


