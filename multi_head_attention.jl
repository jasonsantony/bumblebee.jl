# ------------------------------------------------------------------------------
# multi_head_attention.jl
# functions:
#	multi_head_attention ⧄⧄
# 	scaled_dot_product_attention ✓⧄
#	causal_mask ✓⧄
#	softmax ✓⧄
# 	⊗ (batched matrix multiplication), operator is "\otimes" ✓⧄
# 	batched_transpose ✓⧄
# 	dim_check ✓⧄
# ------------------------------------------------------------------------------
module MultiHeadAttention
	using Einsum # for batched matrix multiplication and transposition

	export scaled_dot_product_attention
	export causal_mask

	# Q: (batch_size, seq_len, d_q)
	# K: (batch_size, seq_len, d_k)
	# V: (batch_size, seq_len, d_V)
	function scaled_dot_product_attention(
		Q::Array{Float32, 3},
		K::Array{Float32, 3},
		V::Array{Float32, 3},
		mask::Union{Array{Float32, 3}, Nothing} = nothing
	)::Array{Float32, 3}
		# make sure Q, K, V, and mask conform
		dim_check(Q, K, V, mask)

		if mask === nothing
			M = zeros(Float32, size(Q, 1), size(Q, 2), size(K, 2)) # (batch_size, seq_len_q, d_v)
		else
			M = mask
		end

		# ------------------------------------------------------------------------------
		# let's spell this out before we get fancy with it:
		# 	1. (Q⊗Kᵀ)/√d_k ⟶ (batch_size, seq_len_q, seq_len_k)
		#   2. add mask
		# 	3. normalize seq_len_k dimension with softmax
		# 	4. Attention(Q,K,V) = softmax((Q⊗Kᵀ)/√d_k + M)⊗V
		#                         ⟶ (batch_size, seq_len_q, d_v)
		# ------------------------------------------------------------------------------

		Kt = batched_transpose(K)
		d_k = size(K, 3)
		sqrt_d_k = Float32(sqrt(d_k))

		# vvv look, just like in the paper! vvv
		return softmax((Q ⊗ Kt ./ sqrt_d_k) + M) ⊗ V

	end



	function causal_mask(
		batch_size::Int,
		seq_len::Int
	)::Array{Float32, 3}
		return reshape(
				[i >= j ? 0.0f0 : -Inf32 
					for b in 1:batch_size
						for i in 1:seq_len
							for j in 1:seq_len
				], 
				batch_size,
				seq_len, seq_len
			)
	end

	function softmax(x::Array{Float32}, dims::Int = 3)::Array{Float32}
		# default dim = 3 for 
		exp_x = exp.(x .- maximum(x, dims=dims))  # Subtract max for numerical stability
		return exp_x ./ sum(exp_x, dims=dims)  # Normalize across the specified dimension
	end

	# batched matrix multiplication
	function batched_matrix_multiplication(A::Array{Float32}, B::Array{Float32})::Array{Float32}
		# Ensure the batch sizes (first dimension) match
		size(A, 1) == size(B, 1) || throw(DimensionMismatch("Batch dimensions must match!"))

		# Check matrix dimensions for compatibility (A's 3rd dimension must match B's 2nd dimension)
		size(A, 3) == size(B, 2) || throw(DimensionMismatch("Matrix dimensions must match: A's columns must match B's rows!"))

		C = @einsum C[b, i, j] := A[b, i, k] * B[b, k, j]
		
		return C
	end
	⊗(A, B) = batched_matrix_multiplication(A, B)

	# batched matrix transposition
	function batched_transpose(A::Array{Float32})::Array{Float32}
		At = @einsum At[b, i, j] := A[b, j, i]
		return At
	end

	function dim_check(
		Q::Array{Float32, 3},
		K::Array{Float32, 3},
		V::Array{Float32, 3},
		mask::Union{Array{Float32, 3}, Nothing}
	)
		if !(size(Q) == size(K))
			throw(
				ArgumentError(
					"Shape mismatch: Q and K must have the same shape. " * 
					"Got sizes: $(size(Q)) and $(size(K))."
				)
			)
		elseif (size(Q, 1) != size(V, 1) || size(Q, 2) != size(V, 2))
			throw(
				ArgumentError(
					"Shape mismatch: V must have the same batch_size and " *
					"seq_length as the other matrices. " *
					"Got sizes: Q:$(size(Q)), K:$(size(K)) and V:$(size(V))."
				)
			)
		elseif mask !== nothing && size(mask) != (size(Q, 1), size(Q, 2), size(K, 2))
			throw(
				ArgumentError(
					"Shape mismatch: mask must have dims (batch_size, seq_len, seq_len)" *
					"Got size: $(size(mask))" *
					"Whereas Q: $(size(Q))"
				)
			)
		end
	end
end