# ------------------------------------------------------------------------------
# multi_head_attention.jl
# structs:
#	MultiHeadAttention
# functions:
#	multi_head_attention ⧄⧄
# 	scaled_dot_product_attention ✓✓
#	causal_mask ✓✓
#	softmax ✓✓
# 	⊗ (batched matrix multiplication), operator is "\otimes" ✓✓
# 	batched_transpose ✓✓
# 	dim_check ✓✓
# ------------------------------------------------------------------------------
module MultiHeadAttentionModule
	using LinearAlgebra: triu # for causal_mask, from std lib

	export scaled_dot_product_attention
	export causal_mask

	# TODO: figure out dimensionality
	mutable struct MultiHeadAttention
		W_q::Matrix{Float32}
		W_k::Matrix{Float32}
		W_v::Matrix{Float32}
		W_o::Matrix{Float32}
		causal_mask::Array{Float32, 3}
		n_heads::Int
		head_dim::Int
	end

	# One head of self-attention
	function scaled_dot_product_attention(
		Q::Array{Float32, 3},
		K::Array{Float32, 3},
		V::Array{Float32, 3},
		causal_mask::Array{Float32, 3}
	)::Array{Float32, 3}
		# Q: (batch_size, seq_len, d_q)
		# K: (batch_size, seq_len, d_k)
		# V: (batch_size, seq_len, d_V)

		# make sure Q, K, V, and mask conform
		dim_check(Q, K, V, causal_mask)

		# ------------------------------------------------------------------------------
		# let's spell this out before we get fancy with it:
		# 	1. (Q⊗Kᵀ)/√d_k ⟶ (batch_size, seq_len_q, seq_len_k)
		#   2. add mask
		# 	3. normalize seq_len_k dimension with softmax
		# 	4. Attention(Q,K,V) = softmax((Q⊗Kᵀ)/√d_k + M)⊗V
		#                         ⟶ (batch_size, seq_len_q, d_v)
		# ------------------------------------------------------------------------------

		Kt = batched_transpose(K)
		d_k = Float32(size(K, 3))
		M = causal_mask

		# vvv look, just like in the paper! vvv
		return softmax((Q ⊗ Kt ./ sqrt(d_k)) + M) ⊗ V

	end



	function causal_mask(
		batch_size::Int,
		seq_len::Int
	)::Array{Float32, 3}
		mask_2d = triu(fill(-Inf32, (seq_len, seq_len)), 1)
		mask_3d = Array{Float32}(undef, batch_size, seq_len, seq_len)
		for i in 1:batch_size
			mask_3d[i, :, :] = mask_2d
		end
		return mask_3d
	end

	function softmax(x::Array{Float32}, dims::Int = 3)::Array{Float32}
		# default dim = 3 for 
		exp_x = exp.(x .- maximum(x, dims=dims))  # Subtract max for numerical stability
		return exp_x ./ sum(exp_x, dims=dims)  # Normalize across the specified dimension
	end

	# batched matrix multiplication, *first dim is batch dim*
	function batched_matrix_multiplication(A::Array{Float32}, B::Array{Float32})::Array{Float32}
		# Ensure the batch sizes (first dimension) match
		size(A, 1) == size(B, 1) || throw(DimensionMismatch("Batch dimensions must match."))

		# Check matrix dimensions for compatibility (A's 3rd dimension must match B's 2nd dimension)
		size(A, 3) == size(B, 2) || throw(DimensionMismatch("Matrix dimensions must match: A's columns must match B's rows."))

		b, n = size(A)[1:2] # A: (b, n, m)
		p = size(B)[3] # B: (b, m, p)
		C = Array{Float32, 3}(undef, b, n, p)
		for i in 1:b
			C[i, :, :] = A[i, :, :] * B[i, :, :]
		end
		return C
	end
	⊗(A, B) = batched_matrix_multiplication(A, B)

	# batched matrix transposition, *first dim is batch dim*
	function batched_transpose(A::Array{Float32})::Array{Float32}
		At = permutedims(A, (1, 3, 2))
		return At
	end

	function dim_check(
		Q::Array{Float32, 3},
		K::Array{Float32, 3},
		V::Array{Float32, 3},
		mask::Array{Float32, 3}
	)
		size(Q) == size(K) == size(V) || throw(
			ArgumentError(
				"Shape mismatch: Q, K, and V must have the same dimensions. " *
				"Got sizes: Q: $(size(Q)), K: $(size(K)), and V: $(size(V))"
			)
		)
		
		size(mask) == (size(Q, 1), size(Q, 2), size(K, 2)) || throw(
			ArgumentError(
				"Shape mismatch: mask must have dims (batch_size, seq_len, seq_len)" *
				"Got size: $(size(mask))" *
				"whereas Q: $(size(Q))."
			)
		)
	end
end