import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math


class Linear(nn.Module):
    """
    Linear transformation module that performs y = Wx without bias.
    
    This implementation follows the interface of PyTorch's nn.Linear but without bias.
    Uses truncated normal initialization as specified in the assignment.
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.
        
        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output  
            device (torch.device, optional): Device to store the parameters on
            dtype (torch.dtype, optional): Data type of the parameters
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight parameter with shape (out_features, in_features)
        # Note: We store W (not W^T) for memory ordering reasons
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        # σ² = 2/(d_in + d_out), truncated at [-3σ, 3σ]
        std = (2.0 / (in_features + out_features)) ** 0.5
        with torch.no_grad():
            nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x (torch.Tensor): Input tensor with shape (..., in_features)
            
        Returns:
            torch.Tensor: Output tensor with shape (..., out_features)
        """
        # Perform matrix multiplication: y = xW^T
        # Since we store W as (out_features, in_features), we need to transpose it
        return torch.matmul(x, self.weight.t())


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):

        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype)
        )

        # Initialize weights using truncated normal distribution
        with torch.no_grad():
            nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)
        

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x/rms * self.weight

        return result.to(in_dtype)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # Round d_ff to nearest multiple of 64
        # self.d_ff = int(round(d_model * 8 / 3.0 / 64)) * 64

        self.weight_w1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.weight_w2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype))
        self.weight_w3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))


        """Initialize weights using truncated normal distribution"""
        std = (2.0 / (self.d_model + self.d_ff)) ** 0.5
        with torch.no_grad():
            nn.init.trunc_normal_(self.weight_w1, mean=0.0, std=std, a=-3*std, b=3*std)
            nn.init.trunc_normal_(self.weight_w2, mean=0.0, std=std, a=-3*std, b=3*std)
            nn.init.trunc_normal_(self.weight_w3, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = torch.matmul(x, self.weight_w1.t())
        swish = output * torch.sigmoid(output)
        gates = torch.matmul(x, self.weight_w3.t())
        swiglu = torch.matmul(swish * gates, self.weight_w2.t())

        return swiglu



class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation with precomputed cos/sin buffers.
    
    This implementation follows the PDF guidance to precompute cos(θi,k) and sin(θi,k) 
    values during initialization and reuse them across layers and batches for efficiency.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.dim_pairs = d_k // 2

        # Compute inverse frequencies: 1 / (theta^(2k/d_k)) for k = 0, 1, ..., d_k/2-1
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        
        # Precompute cos and sin values for all positions up to max_seq_len
        # This follows the PDF guidance to reuse values across layers and batches
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # (max_seq_len, 1)
        freqs = positions * inv_freq.unsqueeze(0)  # (max_seq_len, dim_pairs)

        # Create precomputed cos and sin buffers
        # Using persistent=False as recommended in the PDF since these are derived values
        self.register_buffer('cos_cached', torch.cos(freqs), persistent=False)  # (max_seq_len, dim_pairs)
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)  # (max_seq_len, dim_pairs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor using precomputed cos/sin values.
        
        Args:
            x: Input tensor with shape (..., seq_len, d_k)
            token_positions: Token positions with shape (..., seq_len)
            
        Returns:
            Rotated tensor with same shape as input
        """
        seq_len = x.shape[-2]
        batch_dims = x.shape[:-2]

        # Slice precomputed cos/sin values based on token positions
        # token_positions: (..., seq_len) -> need to index into cached values
        # We need to handle arbitrary batch dimensions, so we'll use advanced indexing


        if len(token_positions.shape) < len(x.shape)-1:
            # Expand token_positions to match batch dimensions
            # for _ in range(len(batch_dims)):
            token_positions = token_positions.unsqueeze(0)
        token_positions = token_positions.expand(*batch_dims, seq_len).contiguous()
             
        # Flatten token_positions for indexing, then reshape back
        flat_positions = token_positions.view(-1)  # (batch_size * seq_len,)

        # Index into precomputed buffers
        cos_emb = self.cos_cached[flat_positions]  # (batch_size * seq_len, dim_pairs)
        sin_emb = self.sin_cached[flat_positions]  # (batch_size * seq_len, dim_pairs)
        
        # Reshape back to match input batch dimensions
        cos_emb = cos_emb.view(*batch_dims, seq_len, self.dim_pairs)  # (..., seq_len, dim_pairs)
        sin_emb = sin_emb.view(*batch_dims, seq_len, self.dim_pairs)  # (..., seq_len, dim_pairs)

        # Reshape input tensor to separate even and odd dimensions
        # x: (..., seq_len, d_k) -> (..., seq_len, dim_pairs, 2)
        x = x.view(*batch_dims, seq_len, self.dim_pairs, 2)

        # Split into even and odd components
        x_even = x[..., 0]  # (..., seq_len, dim_pairs) - even indices (0, 2, 4, ...)
        x_odd = x[..., 1]   # (..., seq_len, dim_pairs) - odd indices (1, 3, 5, ...)

        # Apply RoPE rotation using precomputed cos/sin values
        # The rotation formula is:
        # x_even_new = x_even * cos - x_odd * sin
        # x_odd_new = x_even * sin + x_odd * cos
        x_even_new = x_even * cos_emb - x_odd * sin_emb
        x_odd_new = x_even * sin_emb + x_odd * cos_emb

        # Recombine even and odd components
        # Stack them back together: (..., seq_len, dim_pairs, 2)
        rotated = torch.stack([x_even_new, x_odd_new], dim=-1)

        # Reshape back to original shape: (..., seq_len, d_k)
        output = rotated.view(*batch_dims, seq_len, self.d_k)

        return output



def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    max_vals = torch.max(in_features, dim=dim, keepdim=True)[0]
    shifted = in_features - max_vals
    
    # Compute exp(shifted_x)
    exp_vals = torch.exp(shifted)
    
    # Compute the sum along the specified dimension
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)
    
    # Return the normalized probabilities
    return exp_vals / sum_exp


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    
    # Get the key dimension for scaling
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T
    # Q shape: (..., queries, d_k)
    # K shape: (..., keys, d_k)
    # scores shape: (..., queries, keys)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Scale by sqrt(d_k) for numerical stability
    scores = scores / math.sqrt(d_k)

    # Apply mask if provided
    # Where mask is False, set scores to negative infinity
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax to get attention weights
    # Softmax along the keys dimension (last dimension)
    attn_weights = softmax(scores, dim=-1)

    # Apply attention weights to values
    # attn_weights shape: (..., queries, keys)
    # V shape: (..., keys, d_v)
    # output shape: (..., queries, d_v)
    output = torch.matmul(attn_weights, V)

    return output



class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model / num_heads
        self.d_v = d_model / num_heads


        self.q_proj_weight = nn.Parameter(
            torch.empty(self.d_model, self.d_model, device=device, dtype=dtype)
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty(self.d_model, self.d_model, device=device, dtype=dtype)
        )
        self.v_proj_weight = nn.Parameter(
            torch.empty(self.d_model, self.d_model, device=device, dtype=dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.d_model, self.d_model, device=device, dtype=dtype)
        )

        # Initialize weights using truncated normal distribution
        with torch.no_grad():
            nn.init.trunc_normal_(self.q_proj_weight, mean=0.0, std=1.0, a=-3, b=3)
            nn.init.trunc_normal_(self.k_proj_weight, mean=0.0, std=1.0, a=-3, b=3)
            nn.init.trunc_normal_(self.v_proj_weight, mean=0.0, std=1.0, a=-3, b=3)
            nn.init.trunc_normal_(self.o_proj_weight, mean=0.0, std=1.0, a=-3, b=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Get dimensions
        batch_dims = x.shape[:-2]  # All dimensions except sequence_length and d_in
        seq_len = x.shape[-2]
        d_head = self.d_model // self.num_heads

        # Step 1: Project to Q, K, V using the concatenated weights
        # The weights contain all heads concatenated, so this gives us all heads at once
        Q = torch.matmul(x, self.q_proj_weight.T)  # (..., seq_len, d_model)
        K = torch.matmul(x, self.k_proj_weight.T)  # (..., seq_len, d_model)
        V = torch.matmul(x, self.v_proj_weight.T)  # (..., seq_len, d_model)

        # Step 2: Reshape to separate heads
        # From (..., seq_len, d_model) to (..., seq_len, num_heads, d_head)
        Q = Q.view(*batch_dims, seq_len, self.num_heads, d_head)
        K = K.view(*batch_dims, seq_len, self.num_heads, d_head)
        V = V.view(*batch_dims, seq_len, self.num_heads, d_head)

        # Step 3: Transpose to put heads dimension before sequence dimension
        # From (..., seq_len, num_heads, d_head) to (..., num_heads, seq_len, d_head)
        Q = Q.transpose(-3, -2)  # (..., num_heads, seq_len, d_head)
        K = K.transpose(-3, -2)  # (..., num_heads, seq_len, d_head)
        V = V.transpose(-3, -2)  # (..., num_heads, seq_len, d_head)

        # Step 4: Create causal mask for self-attention
        # Each token should only attend to previous tokens (including itself)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        causal_mask = ~causal_mask  # Invert so True means "attend to"

        # Expand mask for all batch dimensions and heads
        # We need to match the shape of Q, K, V which is (..., num_heads, seq_len, d_head)
        # But for attention, we need (..., num_heads, seq_len, seq_len)
        expanded_mask = causal_mask
        # Add dimensions to match (..., num_heads, seq_len, seq_len)
        for _ in range(len(batch_dims)):
            expanded_mask = expanded_mask.unsqueeze(0)  # Add batch dimensions
        expanded_mask = expanded_mask.unsqueeze(-3)  # Add head dimension: (..., 1, seq_len, seq_len)
        # Now expand to match all heads
        expanded_mask = expanded_mask.expand(*batch_dims, self.num_heads, seq_len, seq_len)
    
        # Step 5: Apply scaled dot-product attention with causal mask
        H = scaled_dot_product_attention(Q, K, V, expanded_mask)  # (..., num_heads, seq_len, d_head)

        # Step 6: Transpose back and reshape to concatenate heads
        # From (..., num_heads, seq_len, d_head) to (..., seq_len, num_heads, d_head)
        H = H.transpose(-3, -2)
        # From (..., seq_len, num_heads, d_head) to (..., seq_len, d_model)
        H = H.contiguous().view(*batch_dims, seq_len, self.d_model)
        
        # Step 7: Apply output projection
        output = torch.matmul(H, self.o_proj_weight.T)  # (..., seq_len, d_model)
        
        return output


class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj_weight = nn.Parameter(
            torch.empty(self.d_model, self.d_model, device=device, dtype=dtype)
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty(self.d_model, self.d_model, device=device, dtype=dtype)
        )
        self.v_proj_weight = nn.Parameter(
            torch.empty(self.d_model, self.d_model, device=device, dtype=dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.d_model, self.d_model, device=device, dtype=dtype)
        )

        # Initialize weights using truncated normal distribution
        with torch.no_grad():
            nn.init.trunc_normal_(self.q_proj_weight, mean=0.0, std=1.0, a=-3, b=3)
            nn.init.trunc_normal_(self.k_proj_weight, mean=0.0, std=1.0, a=-3, b=3)
            nn.init.trunc_normal_(self.v_proj_weight, mean=0.0, std=1.0, a=-3, b=3)
            nn.init.trunc_normal_(self.o_proj_weight, mean=0.0, std=1.0, a=-3, b=3)
        
        # Initialize RoPE with proper parameters
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # Get dimensions
        batch_dims = x.shape[:-2]  # All dimensions except sequence_length and d_in
        seq_len = x.shape[-2]

        # If token_positions not provided, use sequential positions
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        # Step 1: Project to Q, K, V using the concatenated weights
        Q = torch.matmul(x, self.q_proj_weight.T)  # (..., seq_len, d_model)
        K = torch.matmul(x, self.k_proj_weight.T)  # (..., seq_len, d_model)
        V = torch.matmul(x, self.v_proj_weight.T)  # (..., seq_len, d_model)

        # Step 2: Reshape to separate heads
        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.d_head)
        K = K.view(*batch_dims, seq_len, self.num_heads, self.d_head)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.d_head)

        # Step 3: Transpose to put heads dimension before sequence dimension
        Q = Q.transpose(-3, -2)  # (..., num_heads, seq_len, d_head)
        K = K.transpose(-3, -2)  # (..., num_heads, seq_len, d_head)
        V = V.transpose(-3, -2)  # (..., num_heads, seq_len, d_head)

        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Step 5: Create causal mask for self-attention
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        causal_mask = ~causal_mask  # Invert so True means "attend to"

        # Expand mask for all batch dimensions and heads
        expanded_mask = causal_mask
        for _ in range(len(batch_dims)):
            expanded_mask = expanded_mask.unsqueeze(0)  # Add batch dimensions
        expanded_mask = expanded_mask.unsqueeze(-3)  # Add head dimension
        expanded_mask = expanded_mask.expand(*batch_dims, self.num_heads, seq_len, seq_len)
    
        # Step 6: Apply scaled dot-product attention with causal mask using RoPE-rotated Q and K
        H = scaled_dot_product_attention(Q, K, V, expanded_mask)  # (..., num_heads, seq_len, d_head)

        # Step 7: Transpose back and reshape to concatenate heads
        H = H.transpose(-3, -2)
        H = H.contiguous().view(*batch_dims, seq_len, self.d_model)
        
        # Step 8: Apply output projection
        output = torch.matmul(H, self.o_proj_weight.T)  # (..., seq_len, d_model)
        
        return output
    


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()

        self.d_modle = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len)
        self.mha = MultiheadSelfAttentionRoPE(d_model=d_model, num_heads=num_heads, rope=self.rope ,device=device, dtype=dtype)
        self.rmsnorm_1 = RMSNorm(d_model=d_model)
        self.rmsnorm_2 = RMSNorm(d_model=d_model)
        self.ff = SwiGLU(d_model, d_ff)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:

        x_norm = self.rmsnorm_1(x)
        attn_output = self.mha(x_norm, token_positions)
        O = x + attn_output

        normed_O = self.rmsnorm_2(O)
        ff_output = self.ff(normed_O)
        H = O + ff_output

        return H


class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, vocab_size: int, context_length: int, num_layers: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        # self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=theta, max_seq_len=self.context_length, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.rmsnorm = RMSNorm(d_model=d_model)
        self.linear = Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)

        for block in self.transformer_blocks:
            x = block(x, token_positions)
        x = self.rmsnorm(x)
        x = self.linear(x)
        return x

































