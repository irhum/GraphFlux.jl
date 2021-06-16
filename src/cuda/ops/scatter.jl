function scatter(data, targets, bins, ⊕)
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