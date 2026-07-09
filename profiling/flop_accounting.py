"""
Analytical FLOP accounting for modded-nanogpt-relu-mult.

Closed-form FLOP counts for the two audited MLP variants in this repo
(relu_squared baseline and elewise_product) and for the full 12-layer GPT
backbone they share. Every formula cites the source line in the training
scripts so a third party can verify the accounting by inspection.

Conventions
-----------
* 1 multiply-accumulate (MAC) = 2 FLOPs.
* A linear layer d_in -> d_out over T tokens costs 2*T*d_in*d_out FLOPs
  forward. Backward costs twice that (grad-input matmul + grad-weight matmul),
  so forward+backward = 3x forward for matmuls.
* "matmul FLOPs" = mm/addmm/bmm/einsum-contraction only. This is exactly what
  torch.profiler's `with_flops=True` counts, so the empirical numbers from
  profile_flops.py are directly comparable to the matmul column here.
* Elementwise FLOPs (ReLU, Hadamard products, rotary, norms, softmax,
  softcap) are tracked in a separate column. They are <1% of the total but
  are reported so the accounting is complete.

Model backbone (identical in both train scripts):
  vocab 50257 padded to 50304, num_layers=12, num_heads=6, head_dim=128
  (attn hidden = 768), model_dim=768, attention skipped at layer index 7
  (11 attention layers), train_seq_len = 48*1024.
  Source: train_gpt.py L627, L326, L395.

Attention FLOPs depend on the sliding-window block mask (train_gpt.py
L441-479), so they are parameterized by the average number of attended kv
positions per query, `kv_per_query`. Three modes are provided:
  * "dense"    : kv_per_query = T          (what SDPA-math / a full T x T bmm
                                            computes; matches the CPU profiler
                                            verification path exactly)
  * "causal"   : kv_per_query = (T+1)/2    (dense causal upper bound for the
                                            real kernel)
  * "windowed" : kv_per_query = W          (pass the average window size to
                                            match a given training step's
                                            block mask)
"""

from dataclasses import dataclass, field


@dataclass
class MLPFlops:
    name: str
    params: int
    fwd_matmul_per_token: int   # mm/addmm FLOPs, forward, per token
    fwd_elementwise_per_token: int
    notes: str = ""

    def fwd(self, tokens: int) -> int:
        return tokens * self.fwd_matmul_per_token

    def fwd_bwd(self, tokens: int) -> int:
        return 3 * self.fwd(tokens)


def mlp_relu_squared(dim: int = 768) -> MLPFlops:
    """train_gpt_mlp_relu_squared.py L358-370 (baseline modded-nanogpt MLP).

    c_fc: dim -> 4*dim, c_proj: 4*dim -> dim, activation relu(x)^2.
    """
    hdim = 4 * dim
    params = dim * hdim + hdim * dim
    matmul = 2 * dim * hdim + 2 * hdim * dim          # = 16*dim^2
    elementwise = 2 * hdim                            # relu + square
    return MLPFlops("relu_squared (baseline)", params, matmul, elementwise,
                    notes=f"hdim={hdim}; 16*d^2 matmul FLOPs/token")


def mlp_elewise_product(dim: int = 768) -> MLPFlops:
    """train_gpt_mlp_elewise_product.py L358-371.

    c_fc_a, c_fc_b: dim -> 4*dim each, c_proj: 4*dim -> dim,
    activation relu(a)*relu(b) (GLU-style).
    """
    hdim = 4 * dim
    params = 2 * dim * hdim + hdim * dim
    matmul = 2 * (2 * dim * hdim) + 2 * hdim * dim    # = 24*dim^2
    elementwise = 3 * hdim                            # 2x relu + hadamard
    return MLPFlops("elewise_product", params, matmul, elementwise,
                    notes=f"hdim={hdim}; 24*d^2 matmul FLOPs/token")


MLP_VARIANTS = {
    "relu_squared": mlp_relu_squared,
    "elewise_product": mlp_elewise_product,
}


@dataclass
class ModelFlops:
    """Full-model accounting, per training sequence of T tokens."""
    tokens: int
    dim: int
    num_layers: int
    num_attn_layers: int
    attn_hdim: int
    vocab_padded: int
    kv_per_query: float
    mlp: MLPFlops
    components: dict = field(default_factory=dict)

    @property
    def fwd_matmul(self) -> int:
        return sum(v["matmul"] for v in self.components.values())

    @property
    def fwd_elementwise(self) -> int:
        return sum(v["elementwise"] for v in self.components.values())

    @property
    def fwd_bwd_matmul(self) -> int:
        return 3 * self.fwd_matmul


def model_flops(mlp_variant: str = "relu_squared",
                tokens: int = 48 * 1024,
                dim: int = 768,
                num_layers: int = 12,
                num_heads: int = 6,
                head_dim: int = 128,
                vocab: int = 50257,
                attn_mode: str = "causal",
                kv_per_query: float | None = None) -> ModelFlops:
    """Forward FLOPs for one sequence of `tokens` tokens (batch = 1 sequence,
    as required by FlexAttention, train_gpt.py L345).

    attn_mode: "dense" | "causal" | "windowed" (see module docstring).
    """
    T = tokens
    hdim = num_heads * head_dim                       # 768; train_gpt.py L330
    vocab_padded = -(-vocab // 128) * 128             # 50304; L414
    attn_layers = num_layers - 1                      # attn skipped at idx 7; L395

    if attn_mode == "dense":
        W = float(T)
    elif attn_mode == "causal":
        W = (T + 1) / 2
    elif attn_mode == "windowed":
        assert kv_per_query is not None, "pass kv_per_query for windowed mode"
        W = float(kv_per_query)
    else:
        raise ValueError(attn_mode)

    mlp = MLP_VARIANTS[mlp_variant](dim)
    c = {}

    # Embedding + 3 value embeddings: gathers, no matmul (L415-418, L484).
    c["embed+value_embeds"] = {"matmul": 0, "elementwise": 0}

    # Per attention layer (train_gpt.py L343-356):
    qkv = 2 * T * dim * 3 * hdim                      # merged QKV, L346
    scores = int(2 * T * W * head_dim * num_heads)    # QK^T
    av = int(2 * T * W * head_dim * num_heads)        # attn @ V
    out_proj = 2 * T * hdim * dim                     # c_proj, L355
    attn_matmul = qkv + scores + av + out_proj
    # rotary (L317-323): 4 mult + 2 add per (q,k) element -> 6*2*T*hdim;
    # qk rms-norm ~4 FLOPs/elem on 2*T*hdim; softmax ~5 FLOPs per score;
    # value-embed lambda mix ~3*T*hdim (L350).
    attn_elem = int(12 * T * hdim + 8 * T * hdim + 5 * T * W * num_heads
                    + 3 * T * hdim)
    c["attention (x%d layers)" % attn_layers] = {
        "matmul": attn_layers * attn_matmul,
        "elementwise": attn_layers * attn_elem,
    }

    c["mlp (x%d layers)" % num_layers] = {
        "matmul": num_layers * T * mlp.fwd_matmul_per_token,
        "elementwise": num_layers * T * mlp.fwd_elementwise_per_token,
    }

    # Residual adds, lambdas, skip connections, norms (L398-410, L493-510):
    # ~2 norms + 3 residual ops per layer on (T, dim) -> ~12*T*dim per layer.
    c["residual+norms"] = {"matmul": 0,
                           "elementwise": num_layers * 12 * T * dim}

    # LM head (L422, L511): dim -> vocab_padded.
    c["lm_head"] = {"matmul": 2 * T * dim * vocab_padded, "elementwise": 0}

    # Logit softcap 30*sigmoid(logits/(7.5*sqrt(d))) (L513): ~4 FLOPs/logit;
    # cross-entropy ~5 FLOPs/logit (L514).
    c["softcap+CE"] = {"matmul": 0, "elementwise": 9 * T * vocab_padded}

    return ModelFlops(T, dim, num_layers, attn_layers, hdim, vocab_padded,
                      W, mlp, c)


def muon_newton_schulz_flops(shapes: list[tuple[int, int]], ns_steps: int = 5) -> int:
    """Optional: Muon's Newton-Schulz orthogonalization cost per step.

    For each 2D grad G (m x n, m<=n after transpose): per NS iteration
    A = X X^T (2m^2 n), B = b*A + c*A@A (2m^3), X = a*X + B@X (2m^2 n).
    """
    total = 0
    for m, n in shapes:
        m, n = min(m, n), max(m, n)
        total += ns_steps * (2 * m * m * n + 2 * m ** 3 + 2 * m * m * n)
    return total


def fmt(n: float) -> str:
    for unit, s in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]:
        if abs(n) >= unit:
            return f"{n / unit:.3f} {s}"
    return f"{n:.0f} "


def print_mlp_table(dim: int, tokens: int):
    print(f"\n=== MLP variants @ dim={dim}, per {tokens} tokens "
          f"(matmul col == what torch.profiler with_flops counts) ===")
    hdr = (f"{'variant':<28}{'params':>12}{'fwd matmul':>14}{'fwd elemwise':>14}"
           f"{'fwd+bwd matmul':>16}  notes")
    print(hdr); print("-" * len(hdr))
    for key in MLP_VARIANTS:
        m = MLP_VARIANTS[key](dim)
        print(f"{m.name:<28}{m.params:>12,}{fmt(m.fwd(tokens)):>14}"
              f"{fmt(tokens * m.fwd_elementwise_per_token):>14}"
              f"{fmt(m.fwd_bwd(tokens)):>16}  {m.notes}")


def print_model_table(mlp_variant: str, tokens: int, dim: int,
                      attn_mode: str, kv_per_query: float | None):
    mf = model_flops(mlp_variant, tokens=tokens, dim=dim,
                     attn_mode=attn_mode, kv_per_query=kv_per_query)
    print(f"\n=== Full model, mlp={mlp_variant}, T={tokens}, "
          f"attn_mode={attn_mode} (kv/query={mf.kv_per_query:.0f}) ===")
    hdr = f"{'component':<28}{'fwd matmul':>14}{'fwd elemwise':>14}{'% of matmul':>12}"
    print(hdr); print("-" * len(hdr))
    for name, v in mf.components.items():
        pct = 100 * v["matmul"] / mf.fwd_matmul if mf.fwd_matmul else 0
        print(f"{name:<28}{fmt(v['matmul']):>14}{fmt(v['elementwise']):>14}{pct:>11.1f}%")
    print("-" * len(hdr))
    print(f"{'TOTAL forward':<28}{fmt(mf.fwd_matmul):>14}{fmt(mf.fwd_elementwise):>14}")
    print(f"{'TOTAL fwd+bwd (3x fwd)':<28}{fmt(mf.fwd_bwd_matmul):>14}")
    return mf


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--tokens", type=int, default=48 * 1024,
                   help="tokens per sequence (train_seq_len, L580)")
    p.add_argument("--attn-mode", choices=["dense", "causal", "windowed"],
                   default="causal")
    p.add_argument("--kv-per-query", type=float, default=None,
                   help="avg attended kv positions per query (windowed mode)")
    a = p.parse_args()

    print_mlp_table(a.dim, a.tokens)
    for variant in MLP_VARIANTS:
        print_model_table(variant, a.tokens, a.dim, a.attn_mode, a.kv_per_query)
