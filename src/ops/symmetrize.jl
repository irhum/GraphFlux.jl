# function indegrees(targets::CuVector, bins)
#     function kernel!(output, targets)
#         tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x 
    
#         @inbounds if tid <= length(targets)
#             @atomic output[targets[tid]] += 1
#         end
#         return
#     end
    
#     output = CUDA.zeros(Int64, bins)

#     threads = 256
#     blocks = ceil(Int, length(targets) / threads)
#     @cuda blocks=blocks threads=threads kernel!(output, targets)

#     return output
# end

function indegrees(targets::CuVector, bins)
    vals = CUDA.zeros(1, length(targets))
    scatter(vals, targets, bins, +)[1, :]
end