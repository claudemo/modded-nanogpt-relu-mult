Here is the initial diff (obtained with the help of o3: https://chatgpt.com/share/686dc974-ddc4-8010-bf1c-77d4e9352929)

```python
# We want to replace

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x
		
# with something like this
		
class MLP(nn.Module): # if our experiments are successful, we'll rename this class
    def __init__(self, dim: int):
        super().__init__()
        r = int(math.ceil(2 * math.sqrt(2*dim))) # it might be a good idea to experiment with gradually increasing values of r
        self.r = r
        pairs = r * (r + 1) // 2                 # unique i ≤ j pairs
        self.c_fc   = CastedLinear(dim, r)
        self.c_proj = CastedLinear(pairs, dim)
        self.c_proj.weight.detach().zero_()
        idx_i, idx_j = torch.triu_indices(r, r, device='cpu')   # moved to GPU in .to() # I don't understand the "device" parameter, if one needs it, or what should be done with it
        self.register_buffer("idx_i", idx_i, persistent=False)
        self.register_buffer("idx_j", idx_j, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (..., dim)    →   (..., dim)
        Works with arbitrary batch / sequence prefix.
        """
        h = F.relu(self.c_fc(x))                # (..., r)

        # Gather only the upper-triangular coordinates
        # Shape: (..., pairs)
        hi = h[..., self.idx_i]                 # (..., pairs)
        hj = h[..., self.idx_j]                 # (..., pairs)
        q  = hi * hj                            # (..., pairs)

        # Project back to the embedding dimension
        return self.c_proj(q)
```

Convergence per step is good (potentially competitive with the original, if we manage to find the right value of **"r"**)

However, it turns out to be not too GPU-friendly. In a multi-GPU setup, a single training step takes several times as long as
the training step for the original.

The second attempt here (with help of Claude 4 Opus and GPT-4o):

```python
class MLP(nn.Module): # if our experiments are successful, we'll rename this class
    def __init__(self, dim: int):
        super().__init__()
        r = int(math.ceil(2 * math.sqrt(2*dim))) # it might be a good idea to experiment with gradually increasing values of r
        self.r = r
        pairs = r * (r + 1) // 2                 # unique i ≤ j pairs
        self.c_fc   = CastedLinear(dim, r)
        self.c_proj = CastedLinear(pairs, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (..., dim)    →   (..., dim)
        Works with arbitrary batch / sequence prefix.
        """
        h = F.relu(self.c_fc(x))                # (..., r)

        # Gather only the upper-triangular coordinates
        # Shape: (..., pairs)
        # Compute outer product and extract upper triangular
        h_outer = torch.einsum('...i,...j->...ij', h, h)  # (..., r, r)
        
        # Extract upper triangular part more efficiently
        mask = torch.triu(torch.ones(self.r, self.r, device=h.device, dtype=torch.bool))
        q = h_outer[..., mask]

        # According to GPT-4o:
        #     h_outer[..., mask] uses advanced indexing: it flattens the last two dims (r, r) and selects only those entries where mask is True, i.e., the upper triangle
        #     Result shape: (..., r*(r+1)/2) — the number of upper-triangular entries.

        # Project back to the embedding dimension
        return self.c_proj(q)
``'
