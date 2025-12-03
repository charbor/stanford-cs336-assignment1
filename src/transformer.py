from math import sqrt
from jaxtyping import Complex, Int
from torch import Tensor, nn
from einops import einsum, rearrange
import torch
from jaxtyping import Bool, Float, Int


class Linear(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        std = sqrt(2 / (dim_in + dim_out))
        data = torch.empty(dim_out, dim_in, device=device, dtype=dtype)
        self.weights: Float[Tensor, " dim_out dim_in"] = nn.Parameter(data=data, requires_grad=True)
        torch.nn.init.trunc_normal_(
            self.weights, mean=0, std=std, a=-3 * std, b=3 * std
        )

    def forward(self, X: Float[Tensor, " ... dim_in"]) -> Float[Tensor, " ... dim_out"]:
        return einsum(X, self.weights, "... dim_in, dim_out dim_in -> ... dim_out")


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        data = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weights: Float[Tensor, " num_embeddings embedding_dim"] = nn.Parameter(
            data=data, requires_grad=True
        )
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        return self.weights(token_ids)


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.gain: Float[Tensor, " d_model"] = torch.nn.Parameter(
            data=torch.ones(d_model, device=device, dtype=dtype), requires_grad=True
        )
        self.eps = eps

    def forward(
        self, X: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = X.dtype
        X = X.to(torch.float32)
        rms: Float[torch.Tensor, "... 1"] = torch.sqrt(
            X.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        result: Float[Tensor, " ... d_model"] = X / rms * self.gain
        return result.to(in_dtype)


class SwiGLU(nn.Module):

    def __init__(self, dim_in: int, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Parameter(data=torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(data=torch.empty(d_model, d_ff))
        self.W3 = nn.Parameter(data=torch.empty(d_ff, d_model))

        for W in [self.W1, self.W2, self.W3]:
            std = sqrt(2 / (W.shape[1] + W.shape[0]))
            torch.nn.init.trunc_normal_(W, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, X: Float[Tensor, " ... dim_in"]) -> Float[Tensor, " ... d_model"]:
        Y1: Float[Tensor, " ... d_ff"] = X @ self.W1.T
        gate: Float[Tensor, " ... d_ff"] = Y1 * torch.sigmoid(Y1)
        Y3: Float[Tensor, " ... d_ff"] = X @ self.W3.T
        result: Float[Tensor, " ... d_model"] = (gate * Y3) @ self.W2.T
        return result


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        inv_freq = 1 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        self.register_buffer("inv_freq", inv_freq)  # [d_k / 2]

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"], 
        token_positions: Int[Tensor, " ... seq_len"], 
    ) -> Float[Tensor, " ... seq_len d_k"]:
        """
        Cast into complex numbers, rotate, and cast back to real numbers
        """
        # outer product of idxs and inv_freqs
        freqs = einsum(
            token_positions,
            self.inv_freq,
            "... seq_len, ... half_d_k -> ... seq_len half_d_k",
        )

        # cast x to complex numbers: [a, b, c, d, ...] -> [a + bi, c + di, ...]
        real_x: Float[Tensor, " ... seq_len half_d_k"] = x[..., ::2]
        imag_x: Float[Tensor, " ... seq_len half_d_k"] = x[..., 1::2]
        z: Complex[Tensor, " ... seq_len half_d_k"] = torch.complex(real_x, imag_x)
    
        # rotational embedding of x
        z *= torch.exp(1j * freqs)  # same shape

        # complex to real: [a + bi, c + di, ...] -> [a, b, c, d, ...]
        x_rotated: Float[Tensor, " ... seq_len d_k"] = torch.stack([z.real, z.imag], dim=-1).flatten(
            start_dim=-2
        )  # [..., seq_len, d_k]

        return x_rotated


def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """result[i] = exp(x_i) / sum(exp(x_i))"""
    x_shifted = x - torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x_shifted)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores /= sqrt(K.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask, -float("inf"))
    probs = softmax(scores, dim=-1)
    return einsum(probs, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadAttention(nn.Module):
    """
    Set d_v = d_k = d_model / num_heads.
    """

    def __init__(self, d_model: int, num_heads: int, pos_encoder: RoPE | None = None):

        super().__init__()

        # dimensions
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        d_v = d_k = d_model // num_heads
        self.d_v = d_v
        self.d_k = d_k

        # weights
        self.W_O = Linear(d_v * num_heads, d_model)
        self.W_KQV = Linear(d_model, (d_v + 2 * d_k) * num_heads)

        self.pos_encoder = pos_encoder

    def forward(
        self,
        X: Float[Tensor, "batch_size seq_len d_model"],
        token_positions: Int[Tensor, "?"] | None = None,
    ):

        # create Q, K, V
        batch_size, seq_len, d_model = X.shape
        KQV: Float[Tensor, "batch_size seq_len 3*d_model"] = self.W_KQV(X)
        K, Q, V = torch.split(
            KQV,
            [
                self.d_k * self.num_heads,
                self.d_k * self.num_heads,
                self.d_v * self.num_heads,
            ],
            dim=-1,
        )
        K = rearrange(
            K,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        Q = rearrange(
            Q,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        V = rearrange(
            V,
            "batch_size seq_len (num_heads d_v) -> batch_size num_heads seq_len d_v",
            num_heads=self.num_heads,
        )

        # postional encodings
        if self.pos_encoder:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=X.device)
            Q = self.pos_encoder(Q, token_positions)
            K = self.pos_encoder(K, token_positions)

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=X.device, dtype=torch.bool), diagonal=1
        )

        heads: Float[Tensor, "batch_size num_heads seq_len d_v"] = (
            scaled_dot_product_attention(Q=Q, V=V, K=K, mask=mask)
        )
        activations = rearrange(
            heads,
            "batch_size num_heads seq_len d_v -> batch_size seq_len (num_heads d_v)",
        )
        return self.W_O(activations)


class TransformerBlock(nn.Module):

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, pos_encoder: RoPE | None = None
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model=d_model)
        self.norm2 = RMSNorm(d_model=d_model)
        self.mha = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, pos_encoder=pos_encoder
        )
        self.feed_forward = SwiGLU(dim_in=d_model, d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len d_model"],
        token_positions: Int[Tensor, "batch_size seq_len"] | None = None,
    ) -> Float[Tensor, "batch_size seq_len d_model"]:
        x: Float[Tensor, "batch_size seq_len d_model"] = x + self.mha(self.norm1(x), token_positions)
        x: Float[Tensor, "batch_size seq_len d_model"] = x + self.feed_forward(self.norm2(x))
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.embedder = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

        assert d_model % num_heads == 0
        rope = RoPE(
            theta=rope_theta, d_k=d_model // num_heads, max_seq_len=seq_len
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model, num_heads=num_heads, d_ff=d_ff, pos_encoder=rope
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model=d_model)
        self.linear = Linear(dim_in = d_model, dim_out=vocab_size)

    def forward(self, X: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        X: Float[Tensor, "batch_size seq_len d_model"] = self.embedder(X)
        for block in self.transformer_blocks:
            X: Float[Tensor, "batch_size seq_len d_model"] = block(X)
        X: Float[Tensor, "batch_size seq_len d_model"] = self.norm(X)
        X: Float[Tensor, "batch_size seq_len vocab_size"] = self.linear(X)
        return X


