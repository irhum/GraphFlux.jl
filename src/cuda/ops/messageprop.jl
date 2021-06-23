function scatter(data::CuMatrix, targets::CuVector, bins, ⊕)
    function kernel!(output, data, targets)
        thread = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
        i = thread % size(output, 1) + 1
        j = thread ÷ size(output, 1) + 1
    
        @inbounds if i <= size(output, 1) && j <= length(targets)
            @atomic output[i, targets[j]] = ⊕(output[i, targets[j]], data[i, j])
        end
        return
    end
    
    # TODO: FIX TYPES
    output = CUDA.zeros(eltype(data), size(data, 1), bins)

    threads = 256
    blocks = ceil(Int, size(output, 1) * length(targets) / threads)
    @cuda blocks=blocks threads=threads kernel!(output, data, targets)

    return output
end

# Defining the fused message-aggregate operation
# function gather(X::CuMatrix, A::CuSparseMatrixHCOO, ⊕)
#     function kernel!(O, X, I, J, V)
#         thread = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
#         i = thread % size(O, 1) + 1
#         j = thread ÷ size(O, 1) + 1
    
#         @inbounds if i <= size(O, 1) && j <= length(J)
#             @atomic O[i, J[j]] = ⊕(O[i, J[j]], X[i, I[j]] * V[j])
#         end
#         return
#     end
    
#     # TODO: FIX TYPES
#     O = CUDA.zeros(size(X, 1), size(A, 2))

#     threads = 256
#     blocks = ceil(Int, size(O, 1) * length(A.nzVal) / threads)
#     @cuda blocks=blocks threads=threads kernel!(O, X, A.rowInd, A.colInd, A.nzVal)

#     return O
# end