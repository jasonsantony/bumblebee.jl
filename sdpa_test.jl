include("multi_head_attention.jl")
using .MultiHeadAttention
import Random

function main()
    batch_size = 5
    seq_len = 4
    d_q = 3
    d_v = 6

    Q = rand(Float32, batch_size, seq_len, d_q)
    K = rand(Float32, batch_size, seq_len, d_q)
    V = rand(Float32, batch_size, seq_len, d_v)
    mask = causal_mask(batch_size, seq_len)

    return scaled_dot_product_attention(Q, K, V)
end

main()