include("multi_head_attention.jl")
using .MultiHeadAttentionModule
import Random

function main()
    batch_size = 5
    seq_len = 4
    d_k = 3

    Q = rand(Float32, batch_size, seq_len, d_k)
    K = rand(Float32, batch_size, seq_len, d_k)
    V = rand(Float32, batch_size, seq_len, d_k)
    mask = causal_mask(batch_size, seq_len)

    return scaled_dot_product_attention(Q, K, V, mask)
end

main()