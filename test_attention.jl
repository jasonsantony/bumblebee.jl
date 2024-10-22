include("multi_head_attention.jl")
using .MultiHeadAttentionModule
import Random

function test()
    batch_size = 5
    seq_len = 4
    d_k = 3

    Q = rand(Float32, batch_size, seq_len, d_k)
    K = rand(Float32, batch_size, seq_len, d_k)
    V = rand(Float32, batch_size, seq_len, d_k)
    mask = causal_mask(batch_size, seq_len)

    return scaled_dot_product_attention(Q, K, V, mask)
end

function pretty_print_tensor(A::AbstractArray; max_elements=6)
    dims = size(A)
    ndims_A = ndims(A)

    if ndims_A == 1  # 1D array
        print("[")
        if length(A) > max_elements
            print(join(A[1:max_elements], ", "), ", ...")
        else
            print(join(A, ", "))
        end
        println("]")
    elseif ndims_A == 2  # 2D array (matrix)
        println("[")
        for i in 1:dims[1]
            row = A[i, :]
            print("  [")
            if length(row) > max_elements
                print(join(row[1:max_elements], ", "), ", ...")
            else
                print(join(row, ", "))
            end
            println("]")
        end
        println("]")
    else  # Higher-dimensional arrays (3D or more)
        b = size(A, 1)
        for i in 1:b
            println("Tensor slice $i of $(ndims_A)-dimensional tensor, shape: $(dims)")
            pretty_print_tensor(A[i, :, :]; max_elements=max_elements)  # Recursively print 2D slices
            if i < b
                println()
            end
        end
    end
end


# pretty_print_tensor(causal_mask(5, 4))
pretty_print_tensor(test())