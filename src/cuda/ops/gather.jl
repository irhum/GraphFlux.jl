function gather(data, sources)
    function kernel!(output, data, sources)
        thread = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1
        i = thread % size(output, 1) + 1
        j = thread ÷ size(output, 1) + 1
    
        @inbounds if i <= size(output, 1) && j <= length(sources)
            output[i, j] = data[i, sources[j]]
        end
        return
    end
    
    # TODO: FIX TYPES
    output = CUDA.zeros(eltype(data), size(data, 1), length(sources))

    threads = 256
    blocks = ceil(Int, size(output, 1) * length(sources) / threads)
    @cuda blocks=blocks threads=threads kernel!(output, data, sources)

    return output
end