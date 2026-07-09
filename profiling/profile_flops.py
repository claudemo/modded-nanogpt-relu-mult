"""
Empirical FLOP profiling for modded-nanogpt-relu-mult, following the
methodology of https://github.com/Quentin-Anthony/torch-profiling-tutorial:

  * torch.profiler with CPU+CUDA activities
  * schedule(wait=1, warmup=1, active=3) so warmup/compile noise is excluded
  * record_function(...) labels around each logical block
  * record_shapes / profile_memory / with_stack, plus with_flops=True so the
    profiler itself counts matmul FLOPs from recorded shapes -- this is the
    empirical side of the formal accounting; the analytical side lives in
    flop_accounting.py and the script cross-checks the two.
  * tensorboard_trace_handler output for detailed trace inspection.

The model classes below are transcribed from train_gpt.py (line references in
comments) with two deliberate, FLOP-neutral or FLOP-documented deviations:
  1. CastedLinear's fp8 path (train only, L295-302) is replaced by the plain
     F.linear path (same matmul shape => same FLOP count).
  2. flex_attention with the sliding-window block mask is replaced by
     F.scaled_dot_product_attention(is_causal=True). The matmul SHAPES the
     profiler sees correspond to dense attention; flop_accounting.py's
     attn_mode ("dense"/"causal"/"windowed") states exactly which count
     applies to which execution, so nothing is hidden by this substitution.

All flags are independent and optional. Defaults: --mode both (MLP microbench
+ full model), --variant <both variants>, --tokens 49152 (= train_seq_len),
--device cuda if available, forward only.

Usage (GPU box):
  python profiling/profile_flops.py                                        # everything, defaults
  python profiling/profile_flops.py --mode mlp --variant elewise_product   # one variant, MLP only
  python profiling/profile_flops.py --mode model --backward                # full model, fwd+bwd
  tensorboard --logdir=./log     # to view traces

CPU smoke test / verification (smaller T; --sdpa-math makes attention
matmuls profiler-countable so the full-model ratio is ~1):
  python profiling/profile_flops.py --tokens 1024 --device cpu --sdpa-math

NOTE: this script is a FLOP audit, not a step-time benchmark. Because of
deviations 1 and 2 below, profiler *timings/memory* will not match the real
training run (fp8 lm_head is faster than plain linear; dense SDPA at T=48K
does far more attention work than the sliding-window flex_attention). Use
the training logs for step-time claims; use this for FLOP accounting.
"""

import argparse
import math
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flop_accounting as fa

# -----------------------------------------------------------------------------
# Model components, transcribed from train_gpt.py (deviations documented above)

def norm(x: Tensor):  # train_gpt.py L287-288
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):  # train_gpt.py L290-304, fp8 path removed
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):  # train_gpt.py L306-323 (verbatim)
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):  # train_gpt.py L325-356; SDPA instead of flex_attention
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor):
        B, T = x.size(0), x.size(1)
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)) \
            .view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)
        else:
            v = lambdas[0] * v
        # DEVIATION: dense causal SDPA in place of block-sparse flex_attention.
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.c_proj(y)


class MLPReluSquared(nn.Module):  # train_gpt_mlp_relu_squared.py L358-370
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class MLPElewiseProduct(nn.Module):  # train_gpt_mlp_elewise_product.py L358-371
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc_a = CastedLinear(dim, hdim)
        self.c_fc_b = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor):
        x1 = F.relu(self.c_fc_a(x))
        x2 = F.relu(self.c_fc_b(x))
        return self.c_proj(x1 * x2)


MLP_CLASSES = {
    "relu_squared": MLPReluSquared,
    "elewise_product": MLPElewiseProduct,
}


class Block(nn.Module):  # train_gpt.py L391-403
    def __init__(self, dim, num_heads, max_seq_len, layer_idx, mlp_cls):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = mlp_cls(dim)

    def forward(self, x, ve, x0, lambdas, sa_lambdas):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            with torch.profiler.record_function("Attention"):
                x = x + self.attn(norm(x), ve, sa_lambdas)
        with torch.profiler.record_function("MLP"):
            x = x + self.mlp(norm(x))
        return x


def next_multiple_of_n(v, *, n):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class ProfilingGPT(nn.Module):  # train_gpt.py L411-515, minus dist/blockmask machinery
    def __init__(self, vocab_size, num_layers, num_heads, model_dim, max_seq_len, mlp_cls):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, max_seq_len, i, mlp_cls) for i in range(num_layers)])
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.detach().zero_()
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
        ]))

    def forward(self, input_seq: Tensor, target_seq: Tensor):
        assert input_seq.ndim == 1
        with torch.profiler.record_function("Embeddings"):
            ve = [value_embed(input_seq) for value_embed in self.value_embeds]
            ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
            x = x0 = norm(self.embed(input_seq)[None])

        skip_connections = []
        n = len(self.blocks) // 2
        skip_weights = self.scalars[:n]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)

        for i in range(len(self.blocks)):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i])
            if i < n:
                skip_connections.append(x)

        with torch.profiler.record_function("LM-Head+Loss"):
            x = norm(x)
            logits = self.lm_head(x).float()
            logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1) ** 0.5))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="sum")
        return loss


# -----------------------------------------------------------------------------
# Profiling harness (schedule + trace handler per the tutorial)

def make_profiler(logdir: str, device: str):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,   # profiler-counted matmul FLOPs (empirical side)
    )


def total_flops(prof) -> int:
    """Sum of profiler-attributed FLOPs across all ops, per active step."""
    return sum(e.flops for e in prof.key_averages()) // 3  # 3 active steps


def run_steps(prof, fn, steps=5):
    for _ in range(steps):
        fn()
        prof.step()


def profile_mlp_variants(args):
    device = args.device
    T, dim = args.tokens, args.dim
    x = torch.randn(1, T, dim, device=device,
                    dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    variants = [args.variant] if args.variant else list(MLP_CLASSES)
    results = {}
    for name in variants:
        mlp = MLP_CLASSES[name](dim).to(device)
        if device == "cuda":
            mlp = mlp.bfloat16()
        analytical = fa.MLP_VARIANTS[name](dim)

        def step():
            if args.backward:
                x_in = x.detach().requires_grad_(True)
                with torch.profiler.record_function(f"MLP-{name}-fwd"):
                    out = mlp(x_in)
                with torch.profiler.record_function(f"MLP-{name}-bwd"):
                    out.sum().backward()
            else:
                with torch.no_grad(), torch.profiler.record_function(f"MLP-{name}-fwd"):
                    mlp(x)
            if device == "cuda":
                torch.cuda.synchronize()

        with make_profiler(f"{args.logdir}/mlp_{name}", device) as prof:
            run_steps(prof, step)

        measured = total_flops(prof)
        expected = analytical.fwd_bwd(T) if args.backward else analytical.fwd(T)
        results[name] = (measured, expected)
        print(f"\n===== MLP variant: {name}  (T={T}, dim={dim}, "
              f"{'fwd+bwd' if args.backward else 'fwd'}) =====")
        print(prof.key_averages().table(
            sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total",
            row_limit=12))
        print(f"profiler-counted matmul FLOPs / step : {fa.fmt(measured)}FLOPs")
        print(f"analytical matmul FLOPs / step       : {fa.fmt(expected)}FLOPs")
        if expected:
            print(f"ratio (measured / analytical)        : {measured / expected:.4f}")
        print(f"analytical elementwise (not counted by profiler): "
              f"{fa.fmt(T * analytical.fwd_elementwise_per_token)}FLOPs  [{analytical.notes}]")
    return results


def profile_full_model(args):
    device = args.device
    T = args.tokens
    variants = [args.variant] if args.variant else list(MLP_CLASSES)
    for name in variants:
        model = ProfilingGPT(vocab_size=50257, num_layers=12, num_heads=6,
                             model_dim=args.dim, max_seq_len=T,
                             mlp_cls=MLP_CLASSES[name]).to(device)
        if device == "cuda":
            model = model.bfloat16()
        input_seq = torch.randint(0, 50257, (T,), device=device)
        target_seq = torch.randint(0, 50257, (T,), device=device)

        def step():
            loss = model(input_seq, target_seq)
            if args.backward:
                loss.backward()
                model.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()

        ctx = torch.enable_grad() if args.backward else torch.no_grad()
        # --sdpa-math forces the decomposed (bmm) attention so the profiler
        # counts the scores/AV matmuls too -> full-model ratio should be ~1.
        # Otherwise fused SDPA kernels (CPU-flash or CUDA-flash) report no
        # FLOPs and the "(analytical - SDPA)" ratio is the one to check.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdpa_ctx = sdpa_kernel([SDPBackend.MATH]) if args.sdpa_math else nullcontext()
        with make_profiler(f"{args.logdir}/model_{name}", device) as prof:
            with ctx, sdpa_ctx:
                run_steps(prof, step)

        measured = total_flops(prof)
        # SDPA-math on CPU materializes the full T x T bmm -> compare against
        # "dense". On CUDA, flash/mem-efficient SDPA kernels report no FLOPs
        # to the profiler, so subtract nothing but expect the attention
        # scores/AV share to be missing from `measured` (see README).
        mf = fa.model_flops(name, tokens=T, dim=args.dim, attn_mode="dense")
        expected = mf.fwd_bwd_matmul if args.backward else mf.fwd_matmul
        sdpa_share = (3 if args.backward else 1) * mf.num_attn_layers \
            * int(4 * T * mf.kv_per_query * 128 * 6)

        print(f"\n===== Full model, mlp={name}  (T={T}, "
              f"{'fwd+bwd' if args.backward else 'fwd'}) =====")
        print(prof.key_averages().table(
            sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total",
            row_limit=15))
        print(f"profiler-counted matmul FLOPs / step          : {fa.fmt(measured)}FLOPs")
        print(f"analytical matmul FLOPs (attn dense, as run)  : {fa.fmt(expected)}FLOPs")
        print(f"  of which SDPA scores+AV (uncounted on CUDA) : {fa.fmt(sdpa_share)}FLOPs")
        if expected:
            print(f"ratio measured/analytical (~1 with --sdpa-math)       : "
                  f"{measured / expected:.4f}")
            print(f"ratio measured/(analytical - SDPA) (~1 w/ fused SDPA) : "
                  f"{measured / (expected - sdpa_share):.4f}")


def main():
    p = argparse.ArgumentParser(description="FLOP profiling for modded-nanogpt-relu-mult")
    p.add_argument("--mode", choices=["mlp", "model", "both"], default="both")
    p.add_argument("--variant", choices=list(MLP_CLASSES), default=None,
                   help="profile a single MLP variant (default: both)")
    p.add_argument("--tokens", type=int, default=48 * 1024,
                   help="sequence length; use e.g. 1024 for CPU smoke tests")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backward", action="store_true", help="profile fwd+bwd")
    p.add_argument("--sdpa-math", action="store_true",
                   help="force decomposed SDPA so attention matmuls are "
                        "profiler-counted (full-model verification)")
    p.add_argument("--logdir", default="./log")
    args = p.parse_args()

    print(f"device={args.device}, tokens={args.tokens}, dim={args.dim}, "
          f"backward={args.backward}")
    torch.manual_seed(0)

    if args.mode in ("mlp", "both"):
        profile_mlp_variants(args)
    if args.mode in ("model", "both"):
        profile_full_model(args)

    print("\nAnalytical reference tables (flop_accounting.py):")
    fa.print_mlp_table(args.dim, args.tokens)
    print(f"\nTraces written to {args.logdir}/ -- view with: "
          f"tensorboard --logdir={args.logdir}")


if __name__ == "__main__":
    main()
